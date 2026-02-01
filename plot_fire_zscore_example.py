"""
plot_fire_zscore_example.py
繪製單場火災的 7 天 VPD z-score 時間序列
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
DOY_WINDOW = 7

# 目標火災: WOOLSEY (知名大火)
FIRE_NAME = "WOOLSEY"
IG_DATE = pd.Timestamp("2018-11-08")
FIRE_LAT = 34.125
FIRE_LON = -118.824

print("=" * 60)
print(f"  火災: {FIRE_NAME}")
print(f"  點火日: {IG_DATE.strftime('%Y-%m-%d')}")
print(f"  位置: ({FIRE_LAT}, {FIRE_LON})")
print("=" * 60)

# =========================================================
# 1. 載入 VPD 資料
# =========================================================
print("\n[1/4] 載入 VPD 資料...")
vpd_files = sorted(VPD_DIR.glob('*.nc'))
vpd_all = xr.concat([xr.open_dataset(f)['vpd'] for f in vpd_files], dim='time').load()
print(f"  VPD shape: {vpd_all.shape}")

# 找最近的格點
lats = vpd_all.latitude.values
lons = vpd_all.longitude.values
lat_idx = np.argmin(np.abs(lats - FIRE_LAT))
lon_idx = np.argmin(np.abs(lons - FIRE_LON))
print(f"  最近格點: lat={lats[lat_idx]:.2f}, lon={lons[lon_idx]:.2f}")

# =========================================================
# 2. 計算該格點的 DOY 閾值 (μ, σ)
# =========================================================
print("\n[2/4] 計算 DOY 閾值...")
times = pd.to_datetime(vpd_all.time.values)
doys = times.dayofyear

# 該格點的所有歷史資料
vpd_pixel = vpd_all.values[:, lat_idx, lon_idx]

# 計算每個 DOY 的 μ 和 σ
vpd_mu = np.zeros(366)
vpd_sigma = np.zeros(366)

for doy in range(1, 366):
    doy_diff = np.abs(doys - doy)
    doy_diff = np.minimum(doy_diff, 365 - doy_diff)
    mask = doy_diff <= DOY_WINDOW

    vpd_doy = vpd_pixel[mask]
    valid = ~np.isnan(vpd_doy)
    if valid.sum() > 0:
        vpd_mu[doy] = np.nanmean(vpd_doy)
        vpd_sigma[doy] = np.nanstd(vpd_doy)

print("  完成")

# =========================================================
# 3. 提取火災後 7 天的每日 z-score
# =========================================================
print("\n[3/4] 提取 7 天 z-score...")

# 找到點火日之後的 7 天
days_data = []
for day_offset in range(7):
    target_date = IG_DATE + pd.Timedelta(days=day_offset)

    # 找最近的時間點
    time_idx = np.argmin(np.abs(times - target_date))
    actual_date = times[time_idx]

    # 取 VPD 值
    vpd_val = vpd_pixel[time_idx]
    doy = actual_date.dayofyear
    doy_key = min(doy, 365)

    # 計算 z-score
    if vpd_sigma[doy_key] > 0:
        z_score = (vpd_val - vpd_mu[doy_key]) / vpd_sigma[doy_key]
    else:
        z_score = np.nan

    days_data.append({
        'day': day_offset + 1,
        'date': actual_date,
        'vpd': vpd_val,
        'z_score': z_score
    })
    print(f"  Day {day_offset+1}: {actual_date.strftime('%Y-%m-%d')} | VPD={vpd_val:.2f} | z={z_score:.2f}")

df_days = pd.DataFrame(days_data)

# =========================================================
# 4. 繪圖 (類似參考圖)
# =========================================================
print("\n[4/4] 繪製圖表...")

fig, ax = plt.subplots(figsize=(9, 5))

days = df_days['day'].values
z_scores = df_days['z_score'].values

# 繪製折線 (直線連接)
ax.plot(days, z_scores, color='#8B4513', linewidth=2.5, marker='o', markersize=6)

# 1σ 閾值線
ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label='1σ Threshold')

# 填充超過 1σ 的區域
ax.fill_between(days, 1, z_scores, where=(z_scores > 1),
                color='#8B4513', alpha=0.5, interpolate=True)

# 計算累積異常
above_threshold = np.maximum(0, z_scores - 1)
cumulative_anomaly = np.sum(above_threshold)

# 軸設定
ax.set_xlabel('Time (Days)', fontsize=12)
ax.set_ylabel('Daily Climate Variable\n(e.g., VPD)', fontsize=11)
ax.set_title(f'Post-Ignition Window (7 Days)\n{FIRE_NAME} Fire - {IG_DATE.strftime("%Y-%m-%d")}',
             fontsize=12, fontweight='bold')

ax.set_xticks(range(1, 8))
ax.set_xticklabels(['Day 1\n(ignition)'] + [f'Day {i}' for i in range(2, 8)])
ax.set_xlim(0.5, 7.5)

# Y 軸範圍
y_min = min(z_scores.min(), -1) - 0.3
y_max = max(z_scores.max(), 2) + 0.3
ax.set_ylim(y_min, y_max)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 圖例
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('output_golden/Fig_fire_zscore_example.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n已儲存: output_golden/Fig_fire_zscore_example.png")

# 印出統計
print(f"\n=== 統計摘要 ===")
print(f"累積異常 (Σ max(0, z-1)): {cumulative_anomaly:.2f}")
print(f"超過 1σ 的天數: {(z_scores > 1).sum()}")
print(f"最大 z-score: {z_scores.max():.2f}")
