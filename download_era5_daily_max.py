"""
download_era5_daily_v2.py

ERA5 VPD & Wind 下載腳本

計算邏輯:
- VPD: 日平均 (daily mean)，全天 24h，單位 kPa
- Wind: 日最大 (daily max)，全天 24h，單位 m/s

下載策略:
- 季度下載 (避免請求太大被拒)
- 合併變量 (t2m + d2m + u10 + v10 一起下載)
- 並行處理

使用方式:
    python download_era5_daily_v2.py
"""

import time
import calendar
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cdsapi
import numpy as np
import xarray as xr


# =============================================================================
# 設定區
# =============================================================================

YEARS = range(1984, 2025)

# 並行設定
MAX_WORKERS = 10
SLEEP_SECONDS = 0
MAX_RETRIES = 3

# 區域設定 (美國西部)
AREA_US_WEST = [49, -125, 32.5, -114]

# 路徑設定
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "raw_downloads"
VPD_OUT_DIR = PROJECT_ROOT / "ERA5_daily_vpd"
WIND_OUT_DIR = PROJECT_ROOT / "ERA5_daily_sfcWind"

RAW_DIR.mkdir(parents=True, exist_ok=True)
VPD_OUT_DIR.mkdir(parents=True, exist_ok=True)
WIND_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ERA5 設定
SINGLE_DATASET = "reanalysis-era5-single-levels"
GRID = [0.25, 0.25]

# 全天 24 小時
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]

# Tetens 係數 (kPa)
TETENS_A = 0.6108
TETENS_B = 17.27
TETENS_C = 237.3

# Thread-local storage
thread_local = threading.local()
print_lock = threading.Lock()


def safe_print(msg):
    with print_lock:
        print(msg)


def get_client():
    """取得 thread-local CDS client"""
    if not hasattr(thread_local, 'client'):
        thread_local.client = cdsapi.Client(quiet=True)
    return thread_local.client


def get_valid_days_for_months(year: int, months: List[int]) -> List[str]:
    """取得多個月份的合法日期"""
    max_day = max(calendar.monthrange(year, m)[1] for m in months)
    return [f"{d:02d}" for d in range(1, max_day + 1)]


# =============================================================================
# 工具函數
# =============================================================================

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """統一座標名稱"""
    rename_dict = {}
    if "valid_time" in ds.dims:
        rename_dict["valid_time"] = "time"
    if "lat" in ds.coords:
        rename_dict["lat"] = "latitude"
    if "lon" in ds.coords:
        rename_dict["lon"] = "longitude"
    if rename_dict:
        ds = ds.rename(rename_dict)
    if "latitude" in ds.coords:
        if ds.latitude.values[0] > ds.latitude.values[-1]:
            ds = ds.sortby("latitude")
    return ds


# =============================================================================
# 下載函數 - 季度下載 (合併所有變量)
# =============================================================================

def download_with_retry(client, dataset, request, output_path, max_retries=MAX_RETRIES):
    """帶重試的下載函數"""
    for attempt in range(max_retries):
        try:
            client.retrieve(dataset, request, str(output_path))
            time.sleep(SLEEP_SECONDS)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                safe_print(f"      重試 {attempt + 2}/{max_retries}...")
                time.sleep(10)
            else:
                safe_print(f"      失敗: {e}")
                return False
    return False


def download_quarter(year: int, quarter: int) -> Tuple[bool, str]:
    """下載單季資料 (所有變量一起)"""
    months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}[quarter]
    raw_file = RAW_DIR / f"era5_all_{year}_Q{quarter}.nc"

    if raw_file.exists():
        return True, f"[SKIP] {year}_Q{quarter}"

    client = get_client()
    days = get_valid_days_for_months(year, months)

    request = {
        "product_type": "reanalysis",
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": str(year),
        "month": [f"{m:02d}" for m in months],
        "day": days,
        "time": ALL_HOURS,
        "area": AREA_US_WEST,
        "grid": GRID,
        "data_format": "netcdf",
    }

    success = download_with_retry(client, SINGLE_DATASET, request, raw_file)
    status = "OK" if success else "FAIL"
    return success, f"[{status}] {year}_Q{quarter}"


# =============================================================================
# 計算函數
# =============================================================================

def compute_vpd_daily_mean(ds: xr.Dataset) -> xr.DataArray:
    """計算日平均 VPD (kPa)"""
    ds = standardize_coords(ds)

    if "t2m" in ds:
        t2m = ds["t2m"]
    elif "2m_temperature" in ds:
        t2m = ds["2m_temperature"]
    else:
        raise KeyError("找不到 t2m")

    if "d2m" in ds:
        d2m = ds["d2m"]
    elif "2m_dewpoint_temperature" in ds:
        d2m = ds["2m_dewpoint_temperature"]
    else:
        raise KeyError("找不到 d2m")

    T = t2m - 273.15
    Td = d2m - 273.15

    e_s = TETENS_A * np.exp(TETENS_B * T / (T + TETENS_C))
    e_a = TETENS_A * np.exp(TETENS_B * Td / (Td + TETENS_C))

    vpd = (e_s - e_a).clip(min=0)

    if "time" in vpd.dims and vpd.time.size > 31:
        vpd = vpd.resample(time="1D").mean()

    vpd = vpd.rename("vpd")
    vpd.attrs = {
        "long_name": "Daily Mean Vapor Pressure Deficit",
        "units": "kPa",
    }
    return vpd


def compute_wind_daily_max(ds: xr.Dataset) -> xr.DataArray:
    """計算日最大風速"""
    ds = standardize_coords(ds)

    if "u10" in ds:
        u10 = ds["u10"]
    elif "10m_u_component_of_wind" in ds:
        u10 = ds["10m_u_component_of_wind"]
    else:
        raise KeyError("找不到 u10")

    if "v10" in ds:
        v10 = ds["v10"]
    elif "10m_v_component_of_wind" in ds:
        v10 = ds["10m_v_component_of_wind"]
    else:
        raise KeyError("找不到 v10")

    sfcWind = np.sqrt(u10**2 + v10**2)

    if "time" in sfcWind.dims and sfcWind.time.size > 31:
        sfcWind = sfcWind.resample(time="1D").max()

    sfcWind = sfcWind.rename("sfcWind")
    sfcWind.attrs = {
        "long_name": "Daily Maximum 10m Wind Speed",
        "units": "m s-1",
    }
    return sfcWind


# =============================================================================
# 處理函數
# =============================================================================

def process_year(year: int) -> Tuple[bool, bool]:
    """處理單一年份"""
    
    vpd_file = VPD_OUT_DIR / f"vpd_{year}.nc"
    wind_file = WIND_OUT_DIR / f"sfcWind_{year}.nc"
    
    if vpd_file.exists() and wind_file.exists():
        print(f"  [SKIP] {year} 已處理完成")
        return True, True

    # 檢查 4 個季度檔案是否都存在
    raw_files = [RAW_DIR / f"era5_all_{year}_Q{q}.nc" for q in range(1, 5)]
    missing = [f for f in raw_files if not f.exists()]
    
    if missing:
        print(f"  [WAIT] {year}: 缺少 {len(missing)} 個季度檔案")
        return vpd_file.exists(), wind_file.exists()

    print(f"  [PROCESS] {year}...")

    try:
        vpd_list = []
        wind_list = []
        
        for q in range(1, 5):
            raw_file = RAW_DIR / f"era5_all_{year}_Q{q}.nc"
            print(f"    處理 Q{q}...", end=" ", flush=True)
            
            ds_raw = xr.open_dataset(raw_file)
            
            vpd_q = compute_vpd_daily_mean(ds_raw)
            wind_q = compute_wind_daily_max(ds_raw)
            
            vpd_list.append(vpd_q)
            wind_list.append(wind_q)
            
            ds_raw.close()
            print("OK")
        
        # 合併 4 個季度
        print(f"    合併資料...", end=" ", flush=True)
        vpd_year = xr.concat(vpd_list, dim="time")
        wind_year = xr.concat(wind_list, dim="time")
        
        vpd_year = vpd_year.sortby("time")
        wind_year = wind_year.sortby("time")
        
        # 移除重複
        _, unique_idx = np.unique(vpd_year.time, return_index=True)
        vpd_year = vpd_year.isel(time=sorted(unique_idx))
        
        _, unique_idx = np.unique(wind_year.time, return_index=True)
        wind_year = wind_year.isel(time=sorted(unique_idx))
        print("OK")
        
        # 儲存 VPD
        if not vpd_file.exists():
            print(f"    儲存 VPD...", end=" ", flush=True)
            vpd_ds = xr.Dataset(
                data_vars={
                    'vpd': (['time', 'latitude', 'longitude'], vpd_year.values, {
                        "long_name": "Daily Mean Vapor Pressure Deficit",
                        "units": "kPa",
                    })
                },
                coords={
                    'time': vpd_year.time.values,
                    'latitude': vpd_year.latitude.values,
                    'longitude': vpd_year.longitude.values,
                },
                attrs={
                    "title": "ERA5 Daily Mean VPD",
                    "source": "ERA5 reanalysis",
                    "year": str(year),
                }
            )
            vpd_ds.to_netcdf(vpd_file, encoding={
                'vpd': {"zlib": True, "complevel": 4, "dtype": "float32"},
                "time": {"units": "days since 1970-01-01"},
            })
            print(f"OK ({len(vpd_year.time)} days)")

        # 儲存 Wind
        if not wind_file.exists():
            print(f"    儲存 Wind...", end=" ", flush=True)
            wind_ds = xr.Dataset(
                data_vars={
                    'sfcWind': (['time', 'latitude', 'longitude'], wind_year.values, {
                        "long_name": "Daily Maximum 10m Wind Speed",
                        "units": "m s-1",
                    })
                },
                coords={
                    'time': wind_year.time.values,
                    'latitude': wind_year.latitude.values,
                    'longitude': wind_year.longitude.values,
                },
                attrs={
                    "title": "ERA5 Daily Maximum Wind Speed",
                    "source": "ERA5 reanalysis",
                    "year": str(year),
                }
            )
            wind_ds.to_netcdf(wind_file, encoding={
                'sfcWind': {"zlib": True, "complevel": 4, "dtype": "float32"},
                "time": {"units": "days since 1970-01-01"},
            })
            print(f"OK ({len(wind_year.time)} days)")

        return True, True

    except Exception as e:
        print(f"    [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return vpd_file.exists(), wind_file.exists()


# =============================================================================
# 主程式
# =============================================================================

def main():
    print("=" * 70)
    print("  ERA5 Daily VPD & Wind")
    print("=" * 70)
    print()
    print("計算邏輯:")
    print("  VPD: 日平均 (全天 24h) - kPa")
    print("  Wind: 日最大 (全天 24h) - m/s")
    print()
    print("下載策略:")
    print("  - 季度下載 (4 請求/年)")
    print("  - 合併變量 (t2m + d2m + u10 + v10)")
    print(f"  - 並行數: {MAX_WORKERS}")
    print()
    print(f"年份: {min(YEARS)} - {max(YEARS)}")
    print("=" * 70)

    start_time = time.time()

    # 第一階段：並行下載
    print("\n" + "=" * 70)
    print("  階段 1: 下載")
    print("=" * 70)

    # 建立所有下載任務
    tasks = []
    for year in YEARS:
        for quarter in range(1, 5):
            tasks.append((year, quarter))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_quarter, y, q): (y, q) for y, q in tasks}
        
        for future in as_completed(futures):
            success, msg = future.result()
            safe_print(f"  {msg}")

    # 第二階段：處理
    print("\n" + "=" * 70)
    print("  階段 2: 處理")
    print("=" * 70)

    vpd_success = 0
    wind_success = 0

    for year in YEARS:
        vpd_ok, wind_ok = process_year(year)
        if vpd_ok:
            vpd_success += 1
        if wind_ok:
            wind_success += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"完成! 耗時: {elapsed/60:.1f} 分鐘")
    print(f"  VPD: {vpd_success}/{len(list(YEARS))} 年")
    print(f"  Wind: {wind_success}/{len(list(YEARS))} 年")
    print("=" * 70)


if __name__ == "__main__":
    main()