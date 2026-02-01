"""
plot_daily_zscore.py

繪製整區的 z-score 時間序列
- 先計算區域平均
- 再用區域平均的歷史資料計算 μ 和 σ
- 計算每天的 z-score
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# =========================================================
# 設定
# =========================================================

VPD_DIR = Path('/Users/huangzhongchen/Desktop/climate/ERA5_daily_vpd')
WIND_DIR = Path('/Users/huangzhongchen/Desktop/climate/ERA5_daily_sfcWind')
SPEI_FILE = '/Users/huangzhongchen/Desktop/climate/spei12 (1).nc'

LAT_MIN, LAT_MAX = 32.5, 49.0
LON_MIN, LON_MAX = -125.0, -114.0

DOY_WINDOW = 7
START_YEAR = 1984
END_YEAR = 2024

print("=" * 70)
print("  Z-score 時間序列分析")
print("=" * 70)

# =========================================================
# 1. 載入資料
# =========================================================
print("\n[1/3] 載入 VPD...")
vpd_files = sorted(VPD_DIR.glob('*.nc'))
vpd_all = xr.concat([xr.open_dataset(f)['vpd'] for f in vpd_files], dim='time').load()
print(f"  VPD shape: {vpd_all.shape}")

print("\n[2/3] 載入 Wind...")
wind_files = sorted(WIND_DIR.glob('*.nc'))
wind_all = xr.concat([xr.open_dataset(f)['sfcWind'] for f in wind_files], dim='time').load()
print(f"  Wind shape: {wind_all.shape}")

print("\n[3/3] 載入 SPEI-12...")
spei_ds = xr.open_dataset(SPEI_FILE)
spei_all = spei_ds['spei']
spei_region = spei_all.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
print(f"  SPEI shape: {spei_region.shape}")

times = pd.to_datetime(vpd_all.time.values)
doys = times.dayofyear
n_times = len(times)

# =========================================================
# 2. 先計算每天的區域平均
# =========================================================
print("\n  計算每天的區域平均...")

# 每天取整區的空間平均 → 每天一個值
vpd_daily_mean = np.nanmean(vpd_all.values, axis=(1, 2))  # (n_times,)
wind_daily_mean = np.nanmean(wind_all.values, axis=(1, 2))  # (n_times,)
print(f"  每日區域平均 shape: {vpd_daily_mean.shape}")

# =========================================================
# 3. 用區域平均的歷史資料計算 DOY-specific μ 和 σ
# =========================================================
print("\n  計算區域平均的 DOY 閾值...")

vpd_mu = np.zeros(366)
vpd_sigma = np.zeros(366)
wind_mu = np.zeros(366)
wind_sigma = np.zeros(366)

for doy in range(1, 366):
    doy_diff = np.abs(doys - doy)
    doy_diff = np.minimum(doy_diff, 365 - doy_diff)
    mask = doy_diff <= DOY_WINDOW

    vpd_doy = vpd_daily_mean[mask]  # 該 DOY 窗口內的區域平均值
    wind_doy = wind_daily_mean[mask]

    vpd_mu[doy] = np.nanmean(vpd_doy)
    vpd_sigma[doy] = np.nanstd(vpd_doy)
    wind_mu[doy] = np.nanmean(wind_doy)
    wind_sigma[doy] = np.nanstd(wind_doy)

print("  完成")

# =========================================================
# 4. 計算每天的 z-score
# =========================================================
print("\n  計算每日 z-score...")

vpd_zscore_daily = np.zeros(n_times)
wind_zscore_daily = np.zeros(n_times)

for t_idx, doy in enumerate(doys):
    doy_key = min(doy, 365)

    # 當天的區域平均
    vpd_val = vpd_daily_mean[t_idx]
    wind_val = wind_daily_mean[t_idx]

    # z-score = (當天區域平均 - μ_DOY) / σ_DOY
    if vpd_sigma[doy_key] > 0:
        vpd_zscore_daily[t_idx] = (vpd_val - vpd_mu[doy_key]) / vpd_sigma[doy_key]
    else:
        vpd_zscore_daily[t_idx] = np.nan

    if wind_sigma[doy_key] > 0:
        wind_zscore_daily[t_idx] = (wind_val - wind_mu[doy_key]) / wind_sigma[doy_key]
    else:
        wind_zscore_daily[t_idx] = np.nan

    if (t_idx + 1) % 5000 == 0:
        print(f"    進度: {t_idx+1}/{n_times}")

# =========================================================
# 5. 計算 SPEI 月資料 (整區平均)
# =========================================================
print("\n  計算 SPEI...")

spei_times = pd.to_datetime(spei_all.time.values)
spei_monthly = []

for t_idx in range(len(spei_times)):
    spei_t = spei_region.isel(time=t_idx).values
    valid = ~np.isnan(spei_t) & (np.abs(spei_t) < 1e10)
    if valid.sum() > 0:
        spei_monthly.append({
            'time': spei_times[t_idx],
            'spei': np.nanmean(spei_t[valid])
        })

df_spei = pd.DataFrame(spei_monthly)
df_spei = df_spei.set_index('time')

# 只保留 1984-2024 的資料
df_spei = df_spei[(df_spei.index.year >= START_YEAR) & (df_spei.index.year <= END_YEAR)]
print(f"  SPEI 資料筆數: {len(df_spei)}")

# =========================================================
# 5. 建立 DataFrame 並計算移動平均
# =========================================================
print("\n  計算移動平均...")

df_daily = pd.DataFrame({
    'time': times,
    'vpd_z': vpd_zscore_daily,
    'wind_z': wind_zscore_daily
})
df_daily = df_daily.set_index('time')

# SPEI 移動平均 (12個月)
df_spei['spei_rolling'] = df_spei['spei'].rolling(window=12, center=True).mean()
df_spei['drought_rolling'] = (df_spei['spei'] < 0).rolling(window=12, center=True).mean()

# =========================================================
# 6. 計算月平均
# =========================================================
print("\n  計算月平均...")

df_monthly = df_daily[['vpd_z', 'wind_z']].resample('ME').mean()

# =========================================================
# 7. 繪圖 - 三合一 (每日 VPD, Wind + 月 SPEI)
# =========================================================
print("\n  繪製圖表...")

# 篩選 1984-2024 的每日資料
df_daily_filtered = df_daily[(df_daily.index.year >= START_YEAR) & (df_daily.index.year <= END_YEAR)].copy()

fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# 計算滾動百分位數 (5th, 50th, 95th)
window = 365
df_daily_filtered['vpd_p05'] = df_daily_filtered['vpd_z'].rolling(window=window, center=True, min_periods=180).quantile(0.05)
df_daily_filtered['vpd_p50'] = df_daily_filtered['vpd_z'].rolling(window=window, center=True, min_periods=180).quantile(0.50)
df_daily_filtered['vpd_p95'] = df_daily_filtered['vpd_z'].rolling(window=window, center=True, min_periods=180).quantile(0.95)

df_daily_filtered['wind_p05'] = df_daily_filtered['wind_z'].rolling(window=window, center=True, min_periods=180).quantile(0.05)
df_daily_filtered['wind_p50'] = df_daily_filtered['wind_z'].rolling(window=window, center=True, min_periods=180).quantile(0.50)
df_daily_filtered['wind_p95'] = df_daily_filtered['wind_z'].rolling(window=window, center=True, min_periods=180).quantile(0.95)

# 線性趨勢 (對各百分位)
from scipy import stats
x_numeric = np.arange(len(df_daily_filtered))

def calc_trend(y):
    valid = ~np.isnan(y)
    if valid.sum() < 10:
        return np.full_like(y, np.nan), np.nan, np.nan
    slope, intercept, _, p, _ = stats.linregress(x_numeric[valid], y[valid])
    return intercept + slope * x_numeric, p, slope

# VPD 趨勢線
vpd_p05_trend, p_vpd_p05, _ = calc_trend(df_daily_filtered['vpd_p05'].values)
vpd_p50_trend, p_vpd_p50, _ = calc_trend(df_daily_filtered['vpd_p50'].values)
vpd_p95_trend, p_vpd_p95, _ = calc_trend(df_daily_filtered['vpd_p95'].values)

# Wind 趨勢線
wind_p05_trend, p_wind_p05, _ = calc_trend(df_daily_filtered['wind_p05'].values)
wind_p50_trend, p_wind_p50, _ = calc_trend(df_daily_filtered['wind_p50'].values)
wind_p95_trend, p_wind_p95, _ = calc_trend(df_daily_filtered['wind_p95'].values)

# --- VPD (每日) ---
ax = axes[0]
# 每日數據
ax.plot(df_daily_filtered.index, df_daily_filtered['vpd_z'], '-', color='orangered', linewidth=0.3, alpha=0.5)
# 三條趨勢直線
ax.plot(df_daily_filtered.index, vpd_p95_trend, '-', color='darkred', linewidth=2, label='95th trend')
ax.plot(df_daily_filtered.index, vpd_p50_trend, '-', color='black', linewidth=2, label='Median trend')
ax.plot(df_daily_filtered.index, vpd_p05_trend, '-', color='darkred', linewidth=2, alpha=0.7, label='5th trend')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('VPD Z-score', fontsize=11)
ax.set_title('(a) Daily VPD Z-score (1984-2024)', fontsize=12, fontweight='bold')
ax.set_xlim(df_daily_filtered.index[0], df_daily_filtered.index[-1])
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# --- Wind (每日) ---
ax = axes[1]
# 每日數據
ax.plot(df_daily_filtered.index, df_daily_filtered['wind_z'], '-', color='steelblue', linewidth=0.3, alpha=0.5)
# 三條趨勢直線
ax.plot(df_daily_filtered.index, wind_p95_trend, '-', color='darkblue', linewidth=2, label='95th trend')
ax.plot(df_daily_filtered.index, wind_p50_trend, '-', color='black', linewidth=2, label='Median trend')
ax.plot(df_daily_filtered.index, wind_p05_trend, '-', color='darkblue', linewidth=2, alpha=0.7, label='5th trend')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Wind Z-score', fontsize=11)
ax.set_title('(b) Daily Wind Z-score (1984-2024)', fontsize=12, fontweight='bold')
ax.set_xlim(df_daily_filtered.index[0], df_daily_filtered.index[-1])
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# --- SPEI (月，原始：負=乾，正=濕) ---
ax = axes[2]
spei_values = df_spei['spei'].values  # 原始 SPEI
colors = ['brown' if v < 0 else 'teal' for v in spei_values]  # 負值=乾=棕色
ax.bar(df_spei.index, spei_values, color=colors, width=25, alpha=0.7)

# SPEI 線性趨勢
x_spei = np.arange(len(spei_values))
valid_spei = ~np.isnan(spei_values)
slope_spei, intercept_spei, _, p_spei, _ = stats.linregress(x_spei[valid_spei], spei_values[valid_spei])
spei_trend = intercept_spei + slope_spei * x_spei
ax.plot(df_spei.index, spei_trend, '-', color='black', linewidth=2.5, label=f'Trend (p={p_spei:.3f})')

ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.7)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('SPEI-12', fontsize=11)
ax.set_title('(c) Monthly SPEI-12 (1984-2024, negative=dry)', fontsize=12, fontweight='bold')
ax.set_xlim(df_spei.index[0], df_spei.index[-1])
ax.legend(loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig('Fig_zscore_combined.png', dpi=150, bbox_inches='tight')
print("  已儲存: Fig_zscore_combined.png")

print("\n" + "=" * 70)
print("  完成!")
print("=" * 70)
