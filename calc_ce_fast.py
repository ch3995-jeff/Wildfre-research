"""
calc_ce_fast.py - 加速版 Compound Event 計算

使用並行處理加速計算
"""

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 設定區
# =========================================================

FIRE_SHP = '/Users/huangzhongchen/Desktop/climate/mtbs_perimeter_data/mtbs_perims_DD.shp'
OUTPUT_CSV = '/Users/huangzhongchen/Desktop/climate/data_ce_continuous.csv'

VPD_DIR = Path('/Users/huangzhongchen/Desktop/climate/ERA5_daily_vpd')
WIND_DIR = Path('/Users/huangzhongchen/Desktop/climate/ERA5_daily_sfcWind')
SPEI_FILE = '/Users/huangzhongchen/Desktop/climate/spei12 (1).nc'

TARGET_STATES = ['CA', 'WA', 'OR']
VPD_VAR = 'vpd'
WIND_VAR = 'sfcWind'
WINDOWS = [7, 14, 21, 30]
THRESHOLD_C = 1.0
DOY_WINDOW = 7

# 並行設定
N_JOBS = -1  # 使用所有 CPU 核心


def load_climate_data(data_dir, var_name):
    """載入所有氣候資料到記憶體"""
    files = sorted(data_dir.glob('*.nc'))
    datasets = [xr.open_dataset(f)[var_name] for f in files]
    combined = xr.concat(datasets, dim='time')
    # 轉為 numpy array 加速存取
    return combined.load()


def compute_location_climatology(lat_r, lon_r, vpd_all, wind_all):
    """計算單一地點的 climatology（向量化版本）"""
    try:
        # 取得該地點的時間序列
        vpd_ts = vpd_all.sel(latitude=lat_r, longitude=lon_r, method='nearest').values
        wind_ts = wind_all.sel(latitude=lat_r, longitude=lon_r, method='nearest').values
        times = pd.to_datetime(vpd_all.time.values)
        doys = times.dayofyear

        clim = {}

        # 對每個 DOY 計算統計
        for doy in range(1, 366):
            # DOY ± 7 天的遮罩
            doy_diff = np.abs(doys - doy)
            doy_diff = np.minimum(doy_diff, 365 - doy_diff)  # 處理跨年
            mask = doy_diff <= DOY_WINDOW

            vpd_vals = vpd_ts[mask]
            wind_vals = wind_ts[mask]

            vpd_vals = vpd_vals[~np.isnan(vpd_vals)]
            wind_vals = wind_vals[~np.isnan(wind_vals)]

            if len(vpd_vals) > 30 and len(wind_vals) > 30:
                clim[doy] = {
                    'vpd_mu': np.mean(vpd_vals),
                    'vpd_sigma': np.std(vpd_vals),
                    'wind_mu': np.mean(wind_vals),
                    'wind_sigma': np.std(wind_vals),
                }

        return (lat_r, lon_r), clim
    except:
        return (lat_r, lon_r), {}


def compute_fire_metrics(row_data, vpd_all, wind_all, clim_cache, max_window):
    """計算單場火災的所有指標"""
    idx, ig_date, lat, lon = row_data
    lat_r = round(lat, 1)
    lon_r = round(lon, 1)

    result = {'idx': idx}

    try:
        end_date = ig_date + pd.Timedelta(days=max_window - 1)

        # 取得窗口期資料
        vpd_point = vpd_all.sel(latitude=lat_r, longitude=lon_r, method='nearest')
        wind_point = wind_all.sel(latitude=lat_r, longitude=lon_r, method='nearest')

        vpd_window = vpd_point.sel(time=slice(ig_date, end_date)).values
        wind_window = wind_point.sel(time=slice(ig_date, end_date)).values

        if len(vpd_window) < max_window or len(wind_window) < max_window:
            return result

        # 計算每天的 z-score
        z_vpd = np.full(max_window, np.nan)
        z_wind = np.full(max_window, np.nan)

        for day_offset in range(max_window):
            current_date = ig_date + pd.Timedelta(days=day_offset)
            current_doy = current_date.dayofyear

            loc_clim = clim_cache.get((lat_r, lon_r), {})
            if current_doy in loc_clim:
                clim = loc_clim[current_doy]
                if clim['vpd_sigma'] > 0:
                    z_vpd[day_offset] = (vpd_window[day_offset] - clim['vpd_mu']) / clim['vpd_sigma']
                if clim['wind_sigma'] > 0:
                    z_wind[day_offset] = (wind_window[day_offset] - clim['wind_mu']) / clim['wind_sigma']

        # 計算各窗口的指標
        c = THRESHOLD_C
        for w in WINDOWS:
            z_vpd_w = z_vpd[:w]
            z_wind_w = z_wind[:w]

            if np.any(np.isnan(z_vpd_w)) or np.any(np.isnan(z_wind_w)):
                continue

            vpd_excess = np.maximum(0, z_vpd_w - c)
            wind_excess = np.maximum(0, z_wind_w - c)

            result[f'VPD_exp_{w}'] = np.sum(vpd_excess)
            result[f'Wind_exp_{w}'] = np.sum(wind_excess)
            result[f'CE_AND_{w}'] = np.sum(np.sqrt(vpd_excess * wind_excess))
            result[f'CE_OR_{w}'] = np.sum(vpd_excess + wind_excess)
            result[f'CE_days_{w}'] = np.sum((z_vpd_w > c) & (z_wind_w > c))

    except Exception as e:
        pass

    return result


def main():
    print("=" * 70)
    print("  Compound Event 計算 (加速版)")
    print("=" * 70)

    # 1. 讀取火災資料
    print("\n[1/5] 讀取火災資料...")
    gdf = gpd.read_file(FIRE_SHP)
    gdf['State'] = gdf['Event_ID'].str[:2]
    gdf = gdf[gdf['State'].isin(TARGET_STATES)]
    gdf['BurnBndLat'] = gdf['BurnBndLat'].astype(float)
    gdf['BurnBndLon'] = gdf['BurnBndLon'].astype(float)
    gdf['Ig_Date'] = pd.to_datetime(gdf['Ig_Date'])

    df = pd.DataFrame(gdf.drop(columns='geometry'))
    df['Lat'] = df['BurnBndLat']
    df['Lon'] = df['BurnBndLon']
    print(f"  樣本數: {len(df):,}")

    # 2. SPEI12 - 計算低於 -1 的差值
    print("\n[2/5] 處理 SPEI12...")
    spei_ds = xr.open_dataset(SPEI_FILE)
    spei_all = spei_ds['spei'].load()

    df['SPEI12_deficit'] = np.nan  # max(0, -SPEI12)，SPEI12 < 0 即偏乾
    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            spei_val = spei_all.sel(
                time=row['Ig_Date'],
                lat=row['Lat'],
                lon=row['Lon'],
                method='nearest'
            ).values
            if not np.isnan(spei_val) and spei_val < 1e10:
                # SPEI12 < 0 即偏乾
                deficit = max(0, -float(spei_val))
                df.at[idx, 'SPEI12_deficit'] = deficit
        except:
            pass
        if (i + 1) % 1000 == 0:
            print(f"    進度: {i+1}/{len(df)}")

    spei_ds.close()
    print(f"  SPEI12_deficit 有效值: {df['SPEI12_deficit'].notna().sum():,}/{len(df):,}")

    # 3. 載入氣候資料到記憶體
    print("\n[3/5] 載入氣候資料...")
    print("  載入 VPD...")
    vpd_all = load_climate_data(VPD_DIR, VPD_VAR)
    print("  載入 Wind...")
    wind_all = load_climate_data(WIND_DIR, WIND_VAR)

    # 4. 並行計算 Climatology
    print("\n[4/5] 計算 Climatology（並行處理）...")
    df['lat_round'] = df['Lat'].round(1)
    df['lon_round'] = df['Lon'].round(1)
    unique_locs = df[['lat_round', 'lon_round']].drop_duplicates().values.tolist()
    print(f"  唯一地點數: {len(unique_locs)}")

    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(compute_location_climatology)(lat_r, lon_r, vpd_all, wind_all)
        for lat_r, lon_r in unique_locs
    )

    # 建立 cache
    clim_cache = {loc: clim for loc, clim in results if clim}
    print(f"  有效地點: {len(clim_cache)}")

    # 5. 計算火災指標
    print("\n[5/5] 計算 CE 指標（並行處理）...")
    max_window = max(WINDOWS)

    # 準備資料
    fire_data = [
        (idx, pd.Timestamp(row['Ig_Date']), row['Lat'], row['Lon'])
        for idx, row in df.iterrows()
    ]

    # 並行計算
    fire_results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(compute_fire_metrics)(data, vpd_all, wind_all, clim_cache, max_window)
        for data in fire_data
    )

    # 合併結果
    for result in fire_results:
        idx = result.pop('idx')
        for key, val in result.items():
            df.at[idx, key] = val

    # 統計
    print("\n  完成統計:")
    for w in WINDOWS:
        col = f'CE_AND_{w}'
        valid = df[col].notna().sum()
        print(f"    {col}: {valid:,}/{len(df):,} 有效")

    # 儲存
    df = df.drop(columns=['lat_round', 'lon_round'], errors='ignore')
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  已儲存: {OUTPUT_CSV}")
    print(f"  樣本數: {len(df):,}")


if __name__ == "__main__":
    main()
