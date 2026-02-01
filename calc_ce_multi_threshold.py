"""
calc_ce_multi_threshold.py - 計算多個 threshold 的 exposure
threshold = 0σ, 0.5σ, 1σ, 1.5σ, 2σ
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 設定區
# =========================================================

DATA_CSV = '/Users/huangzhongchen/Desktop/climate/data_ce_continuous.csv'
VPD_DIR = Path('/Users/huangzhongchen/Desktop/climate/ERA5_daily_vpd')
WIND_DIR = Path('/Users/huangzhongchen/Desktop/climate/ERA5_daily_sfcWind')

# 多個 threshold
THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0]
WINDOW = 7
DOY_WINDOW = 7
N_JOBS = -1


def load_climate_data(data_dir, var_name):
    files = sorted(data_dir.glob('*.nc'))
    datasets = [xr.open_dataset(f)[var_name] for f in files]
    combined = xr.concat(datasets, dim='time')
    return combined.load()


def compute_location_climatology(lat_r, lon_r, vpd_all, wind_all):
    try:
        vpd_ts = vpd_all.sel(latitude=lat_r, longitude=lon_r, method='nearest').values
        wind_ts = wind_all.sel(latitude=lat_r, longitude=lon_r, method='nearest').values
        times = pd.to_datetime(vpd_all.time.values)
        doys = times.dayofyear

        clim = {}
        for doy in range(1, 366):
            doy_diff = np.abs(doys - doy)
            doy_diff = np.minimum(doy_diff, 365 - doy_diff)
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


def compute_fire_metrics(row_data, vpd_all, wind_all, clim_cache):
    idx, ig_date, lat, lon = row_data
    lat_r = round(lat, 1)
    lon_r = round(lon, 1)
    result = {'idx': idx}

    try:
        end_date = ig_date + pd.Timedelta(days=WINDOW - 1)
        vpd_point = vpd_all.sel(latitude=lat_r, longitude=lon_r, method='nearest')
        wind_point = wind_all.sel(latitude=lat_r, longitude=lon_r, method='nearest')

        vpd_window = vpd_point.sel(time=slice(ig_date, end_date)).values
        wind_window = wind_point.sel(time=slice(ig_date, end_date)).values

        if len(vpd_window) < WINDOW or len(wind_window) < WINDOW:
            return result

        z_vpd = np.full(WINDOW, np.nan)
        z_wind = np.full(WINDOW, np.nan)

        for day_offset in range(WINDOW):
            current_date = ig_date + pd.Timedelta(days=day_offset)
            current_doy = current_date.dayofyear

            loc_clim = clim_cache.get((lat_r, lon_r), {})
            if current_doy in loc_clim:
                clim = loc_clim[current_doy]
                if clim['vpd_sigma'] > 0:
                    z_vpd[day_offset] = (vpd_window[day_offset] - clim['vpd_mu']) / clim['vpd_sigma']
                if clim['wind_sigma'] > 0:
                    z_wind[day_offset] = (wind_window[day_offset] - clim['wind_mu']) / clim['wind_sigma']

        if np.any(np.isnan(z_vpd)) or np.any(np.isnan(z_wind)):
            return result

        # 計算每個 threshold 的 exposure
        for c in THRESHOLDS:
            # 欄位名稱後綴: t00, t05, t10, t15, t20
            suffix = f"t{int(c*10):02d}"

            vpd_excess = np.maximum(0, z_vpd - c)
            wind_excess = np.maximum(0, z_wind - c)

            result[f'VPD_exp_7_{suffix}'] = np.sum(vpd_excess)
            result[f'Wind_exp_7_{suffix}'] = np.sum(wind_excess)
            result[f'CE_AND_7_{suffix}'] = np.sum(np.sqrt(vpd_excess * wind_excess))

    except:
        pass

    return result


def main():
    print("=" * 70)
    print("  Exposure 計算 (多個 Threshold)")
    print(f"  Thresholds: {THRESHOLDS}")
    print("=" * 70)

    # 載入現有資料
    print("\n[1/5] 載入現有資料...")
    df = pd.read_csv(DATA_CSV)
    df['Ig_Date'] = pd.to_datetime(df['Ig_Date'])
    print(f"  樣本數: {len(df):,}")

    # 載入氣候資料
    print("\n[2/5] 載入 VPD...")
    vpd_all = load_climate_data(VPD_DIR, 'vpd')
    print(f"  VPD shape: {vpd_all.shape}")

    print("\n[3/5] 載入 Wind...")
    wind_all = load_climate_data(WIND_DIR, 'sfcWind')
    print(f"  Wind shape: {wind_all.shape}")

    # 計算 Climatology
    print("\n[4/5] 計算 Climatology...")
    df['lat_round'] = df['Lat'].round(1)
    df['lon_round'] = df['Lon'].round(1)
    unique_locs = df[['lat_round', 'lon_round']].drop_duplicates().values.tolist()
    print(f"  唯一地點數: {len(unique_locs)}")

    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(compute_location_climatology)(lat_r, lon_r, vpd_all, wind_all)
        for lat_r, lon_r in unique_locs
    )
    clim_cache = {loc: clim for loc, clim in results if clim}
    print(f"  有效地點: {len(clim_cache)}")

    # 計算 exposure
    print("\n[5/5] 計算 Exposure (所有 thresholds)...")
    fire_data = [
        (idx, pd.Timestamp(row['Ig_Date']), row['Lat'], row['Lon'])
        for idx, row in df.iterrows()
    ]

    fire_results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(compute_fire_metrics)(data, vpd_all, wind_all, clim_cache)
        for data in fire_data
    )

    # 合併結果
    for result in fire_results:
        idx = result.pop('idx')
        for key, val in result.items():
            df.at[idx, key] = val

    # 統計
    print("\n" + "=" * 70)
    print("  統計摘要")
    print("=" * 70)

    for c in THRESHOLDS:
        suffix = f"t{int(c*10):02d}"
        vpd_col = f'VPD_exp_7_{suffix}'
        wind_col = f'Wind_exp_7_{suffix}'
        ce_col = f'CE_AND_7_{suffix}'

        print(f"\n  Threshold = {c}σ ({suffix}):")
        print(f"    VPD:  mean={df[vpd_col].mean():.2f}, >0: {(df[vpd_col]>0).sum():,} ({(df[vpd_col]>0).mean()*100:.1f}%)")
        print(f"    Wind: mean={df[wind_col].mean():.2f}, >0: {(df[wind_col]>0).sum():,} ({(df[wind_col]>0).mean()*100:.1f}%)")
        print(f"    CE:   mean={df[ce_col].mean():.2f}, >0: {(df[ce_col]>0).sum():,} ({(df[ce_col]>0).mean()*100:.1f}%)")

    # 儲存
    df = df.drop(columns=['lat_round', 'lon_round'], errors='ignore')
    df.to_csv(DATA_CSV, index=False)

    print("\n" + "=" * 70)
    print(f"  已儲存: {DATA_CSV}")
    print(f"  新增欄位:")
    for c in THRESHOLDS:
        suffix = f"t{int(c*10):02d}"
        print(f"    - VPD_exp_7_{suffix}, Wind_exp_7_{suffix}, CE_AND_7_{suffix}")
    print("=" * 70)


if __name__ == "__main__":
    main()
