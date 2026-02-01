"""
plot_fire_climate_map.py

繪製西岸三州 (CA, WA, OR) 的火災與氣候地圖：
1. 火災分布 (MTBS, >1000 acres)
2. 平均 VPD×Wind z-score
3. 平均 SPEI-12
"""

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

# =========================================================
# 設定
# =========================================================
STATE_SHP = '/Users/huangzhongchen/Desktop/climate/tl_2025_us_state/tl_2025_us_state.shp'
MTBS_SHP = '/Users/huangzhongchen/Desktop/climate/mtbs_perimeter_data/mtbs_perims_DD.shp'
VPD_DIR = Path('/Users/huangzhongchen/Desktop/climate/ERA5_daily_vpd')
WIND_DIR = Path('/Users/huangzhongchen/Desktop/climate/ERA5_daily_sfcWind')
SPEI_FILE = '/Users/huangzhongchen/Desktop/climate/spei12 (1).nc'

TARGET_STATES = ['CA', 'WA', 'OR']
MIN_FIRE_ACRES = 1000
DOY_WINDOW = 7

print("=" * 70)
print("  火災與氣候地圖")
print("=" * 70)

# =========================================================
# 1. 載入州界
# =========================================================
print("\n[1/6] 載入州界...")
states = gpd.read_file(STATE_SHP)
west_states = states[states['STUSPS'].isin(TARGET_STATES)]
print(f"  載入 {len(west_states)} 個州")

# =========================================================
# 2. 載入火災資料
# =========================================================
print("\n[2/6] 載入火災資料...")
mtbs = gpd.read_file(MTBS_SHP)
mtbs['State'] = mtbs['Event_ID'].str[:2]
mtbs_west = mtbs[mtbs['State'].isin(TARGET_STATES)]
mtbs_west = mtbs_west[mtbs_west['BurnBndAc'] >= MIN_FIRE_ACRES]
mtbs_west['Ig_Date'] = pd.to_datetime(mtbs_west['Ig_Date'])
mtbs_west = mtbs_west[(mtbs_west['Ig_Date'].dt.year >= 1984) & (mtbs_west['Ig_Date'].dt.year <= 2024)]
print(f"  火災數: {len(mtbs_west):,} (>= {MIN_FIRE_ACRES} acres)")

# =========================================================
# 3. 載入 VPD 和 Wind，計算 z-score
# =========================================================
print("\n[3/6] 載入 VPD...")
vpd_files = sorted(VPD_DIR.glob('*.nc'))
vpd_all = xr.concat([xr.open_dataset(f)['vpd'] for f in vpd_files], dim='time').load()

print("\n[4/6] 載入 Wind...")
wind_files = sorted(WIND_DIR.glob('*.nc'))
wind_all = xr.concat([xr.open_dataset(f)['sfcWind'] for f in wind_files], dim='time').load()

# 計算 DOY climatology
print("\n  計算 DOY 閾值...")
times = pd.to_datetime(vpd_all.time.values)
doys = times.dayofyear

vpd_mu = {}
vpd_sigma = {}
wind_mu = {}
wind_sigma = {}

for doy in range(1, 366):
    doy_diff = np.abs(doys - doy)
    doy_diff = np.minimum(doy_diff, 365 - doy_diff)
    mask = doy_diff <= DOY_WINDOW

    vpd_doy = vpd_all.isel(time=mask).values
    wind_doy = wind_all.isel(time=mask).values

    vpd_mu[doy] = np.nanmean(vpd_doy, axis=0)
    vpd_sigma[doy] = np.nanstd(vpd_doy, axis=0)
    wind_mu[doy] = np.nanmean(wind_doy, axis=0)
    wind_sigma[doy] = np.nanstd(wind_doy, axis=0)

# =========================================================
# 4. 載入 SPEI-12 (先載入以便計算三重複合事件)
# =========================================================
print("\n[5/6] 載入 SPEI-12...")
spei_ds = xr.open_dataset(SPEI_FILE)
spei_all = spei_ds['spei']

# 篩選 1984-2024
spei_times_all = pd.to_datetime(spei_all.time.values)
mask_time = (spei_times_all.year >= 1984) & (spei_times_all.year <= 2024)
spei_filtered = spei_all.isel(time=mask_time)
spei_times = spei_times_all[mask_time]

# 將 SPEI 插值到 VPD/Wind 網格
from scipy.interpolate import RegularGridInterpolator
spei_lats = spei_all.lat.values
spei_lons = spei_all.lon.values
lats = vpd_all.latitude.values
lons = vpd_all.longitude.values

# 建立 SPEI 月份索引 (year, month) -> time_idx
spei_month_idx = {}
for i, t in enumerate(spei_times):
    spei_month_idx[(t.year, t.month)] = i

# 計算極端天數比例
print("\n  計算極端天數比例...")
n_times = len(vpd_all.time)
ce_count = np.zeros_like(vpd_all.isel(time=0).values)  # VPD>1σ AND Wind>1σ
ce3_count = np.zeros_like(vpd_all.isel(time=0).values)  # VPD>1σ AND Wind>1σ AND SPEI<0
valid_count = np.zeros_like(vpd_all.isel(time=0).values)

# 預先插值所有月份的 SPEI 到 VPD/Wind 網格
print("  預處理 SPEI 插值...")
spei_interp_cache = {}
lon_grid, lat_grid = np.meshgrid(lons, lats)

for i in range(len(spei_times)):
    t = spei_times[i]
    key = (t.year, t.month)
    spei_month = spei_filtered.isel(time=i).values

    # 處理 NaN
    spei_filled = np.nan_to_num(spei_month, nan=0)

    # 插值
    interp = RegularGridInterpolator(
        (spei_lats, spei_lons), spei_filled,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    spei_interp_cache[key] = interp((lat_grid, lon_grid))

print("  計算每日極端...")
for t_idx, doy in enumerate(doys):
    doy_key = min(doy, 365)
    time_val = times[t_idx]

    vpd_t = vpd_all.isel(time=t_idx).values
    wind_t = wind_all.isel(time=t_idx).values

    with np.errstate(divide='ignore', invalid='ignore'):
        vpd_z = (vpd_t - vpd_mu[doy_key]) / vpd_sigma[doy_key]
        wind_z = (wind_t - wind_mu[doy_key]) / wind_sigma[doy_key]

    # CE: VPD > 1σ AND Wind > 1σ
    ce = (vpd_z > 1) & (wind_z > 1)
    valid = ~np.isnan(vpd_z) & ~np.isnan(wind_z)

    ce_count[valid] += ce[valid].astype(float)
    valid_count[valid] += 1

    # 三重複合事件: VPD > 1σ AND Wind > 1σ AND SPEI < 0
    spei_key = (time_val.year, time_val.month)
    if spei_key in spei_interp_cache:
        spei_grid = spei_interp_cache[spei_key]
        ce3 = ce & (spei_grid < 0)
        ce3_count[valid] += ce3[valid].astype(float)

    if (t_idx + 1) % 5000 == 0:
        print(f"    進度: {t_idx+1}/{n_times}")

# 極端天數比例
ce_ratio = np.where(valid_count > 0, ce_count / valid_count, np.nan)
ce3_ratio = np.where(valid_count > 0, ce3_count / valid_count, np.nan)

# 計算 SPEI < 0 的比例
spei_drought_ratio = (spei_filtered < 0).mean(dim='time').values
spei_lats = spei_all.lat.values
spei_lons = spei_all.lon.values

print(f"  SPEI drought ratio shape: {spei_drought_ratio.shape}")

# =========================================================
# 5. 繪圖
# =========================================================
print("\n[6/6] 繪製地圖...")

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, figure=fig, wspace=0.02, hspace=0.02,
              left=0.08, right=0.92, top=0.98, bottom=0.02)

axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
axes = np.array(axes)

# 繪圖範圍
lon_min, lon_max = -126, -114
lat_min, lat_max = 32, 50

# --- 圖1: 火災分布 ---
ax = axes[0, 0]
west_states.boundary.plot(ax=ax, linewidth=0.5, color='black')
west_states.plot(ax=ax, facecolor='lightgray', edgecolor='black', alpha=0.3)

# 繪製火災點
fire_pts = mtbs_west.copy()
fire_pts['lon'] = fire_pts['BurnBndLon'].astype(float)
fire_pts['lat'] = fire_pts['BurnBndLat'].astype(float)

scatter = ax.scatter(
    fire_pts['lon'], fire_pts['lat'],
    c=fire_pts['BurnBndAc'],
    s=fire_pts['BurnBndAc'] / 800,
    cmap='YlOrRd',
    norm=LogNorm(vmin=1000, vmax=500000),
    alpha=0.6,
    edgecolors='none'
)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xticks([])
ax.set_yticks([])
ax.text(0.02, 0.98, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# 建立陸地遮罩 (只顯示三州內的格點)
from shapely.geometry import Point

# VPD/Wind 網格的陸地遮罩
print("  建立陸地遮罩...")
lon_grid, lat_grid = np.meshgrid(lons, lats)
west_union = west_states.unary_union

land_mask_vpd = np.zeros(lon_grid.shape, dtype=bool)
for i in range(lon_grid.shape[0]):
    for j in range(lon_grid.shape[1]):
        pt = Point(lon_grid[i, j], lat_grid[i, j])
        if west_union.contains(pt):
            land_mask_vpd[i, j] = True

# SPEI 網格的陸地遮罩
spei_lon_grid, spei_lat_grid = np.meshgrid(spei_lons, spei_lats)
land_mask_spei = np.zeros(spei_lon_grid.shape, dtype=bool)
for i in range(spei_lon_grid.shape[0]):
    for j in range(spei_lon_grid.shape[1]):
        pt = Point(spei_lon_grid[i, j], spei_lat_grid[i, j])
        if west_union.contains(pt):
            land_mask_spei[i, j] = True

# 插值到更細的網格
from scipy.interpolate import RegularGridInterpolator

# 高解析度網格
fine_lons = np.linspace(lon_min, lon_max, 500)
fine_lats = np.linspace(lat_min, lat_max, 500)
fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lons, fine_lats)

# CE ratio 插值
print("  插值 CE ratio...")
ce_filled = np.nan_to_num(ce_ratio, nan=0)
interp_ce = RegularGridInterpolator((lats, lons), ce_filled, method='linear', bounds_error=False, fill_value=np.nan)
ce_fine = interp_ce((fine_lat_grid, fine_lon_grid))

# 建立高解析度陸地遮罩
land_mask_fine = np.zeros(fine_lon_grid.shape, dtype=bool)
for i in range(0, fine_lon_grid.shape[0], 5):  # 每5個點檢查一次加速
    for j in range(0, fine_lon_grid.shape[1], 5):
        pt = Point(fine_lon_grid[i, j], fine_lat_grid[i, j])
        if west_union.contains(pt):
            # 填充周圍 5x5 區域
            land_mask_fine[max(0,i-2):min(fine_lon_grid.shape[0],i+3),
                          max(0,j-2):min(fine_lon_grid.shape[1],j+3)] = True

# SPEI drought ratio 插值
print("  插值 SPEI drought ratio...")
spei_filled = np.nan_to_num(spei_drought_ratio, nan=0)
interp_spei = RegularGridInterpolator((spei_lats, spei_lons), spei_filled, method='linear', bounds_error=False, fill_value=np.nan)
spei_fine = interp_spei((fine_lat_grid, fine_lon_grid))

# --- 圖2 (b): SPEI < 0 比例 ---
ax = axes[0, 1]
ax.set_facecolor('white')

spei_plot = np.ma.masked_where(~land_mask_fine, spei_fine)
spei_plot = np.ma.masked_invalid(spei_plot)

im = ax.imshow(
    spei_plot * 100,  # 轉為百分比
    extent=[lon_min, lon_max, lat_min, lat_max],
    origin='lower',
    cmap='YlOrBr',
    vmin=40, vmax=70,
    aspect='auto',
    interpolation='bilinear'
)
west_states.boundary.plot(ax=ax, linewidth=0.5, color='black', zorder=3)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xticks([])
ax.set_yticks([])
ax.text(0.02, 0.98, 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# --- 圖3 (c): CE 極端天數比例 ---
ax = axes[1, 0]
ax.set_facecolor('white')

ce_plot = np.ma.masked_where(~land_mask_fine, ce_fine)
ce_plot = np.ma.masked_invalid(ce_plot)

im = ax.imshow(
    ce_plot * 100,  # 轉為百分比
    extent=[lon_min, lon_max, lat_min, lat_max],
    origin='lower',
    cmap='Purples',
    vmin=0, vmax=5,
    aspect='auto',
    interpolation='bilinear'
)
west_states.boundary.plot(ax=ax, linewidth=0.5, color='black', zorder=3)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xticks([])
ax.set_yticks([])
ax.text(0.02, 0.98, 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# --- 圖4 (d): 三重複合事件 ---
ax = axes[1, 1]
ax.set_facecolor('white')

# 插值 CE3 ratio
print("  插值 CE3 ratio...")
ce3_filled = np.nan_to_num(ce3_ratio, nan=0)
interp_ce3 = RegularGridInterpolator((lats, lons), ce3_filled, method='linear', bounds_error=False, fill_value=np.nan)
ce3_fine = interp_ce3((fine_lat_grid, fine_lon_grid))

ce3_plot = np.ma.masked_where(~land_mask_fine, ce3_fine)
ce3_plot = np.ma.masked_invalid(ce3_plot)

im = ax.imshow(
    ce3_plot * 100,  # 轉為百分比
    extent=[lon_min, lon_max, lat_min, lat_max],
    origin='lower',
    cmap='RdPu',
    vmin=0, vmax=3,
    aspect='auto',
    interpolation='bilinear'
)
west_states.boundary.plot(ax=ax, linewidth=0.5, color='black', zorder=3)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xticks([])
ax.set_yticks([])
ax.text(0.02, 0.98, 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# 添加側邊 colorbar (距離地圖 0.05，左側標籤在左，右側標籤在右)
# 地圖範圍: left=0.08, right=0.92
# 左側 colorbar (火災面積 - 上排左): 0.08 - 0.05 - 0.012 = 0.018
cax1 = fig.add_axes([0.018, 0.52, 0.012, 0.44])
cb1 = plt.colorbar(scatter, cax=cax1)
cb1.ax.yaxis.set_ticks_position('right')
cb1.ax.yaxis.set_label_position('left')
cb1.set_label('Burned area (acres)', fontsize=8, rotation=90, labelpad=5)
cb1.ax.tick_params(labelsize=7)

# 右側 colorbar (乾旱比例 - 上排右): 0.92 + 0.05 = 0.97
cax2 = fig.add_axes([0.97, 0.52, 0.012, 0.44])
cb2 = fig.colorbar(plt.cm.ScalarMappable(cmap='YlOrBr', norm=plt.Normalize(40, 70)), cax=cax2)
cb2.ax.yaxis.set_ticks_position('left')
cb2.ax.yaxis.set_label_position('right')
cb2.set_label('Months drier than the local\nclimatological normal (%)', fontsize=7, rotation=270, labelpad=15)
cb2.ax.tick_params(labelsize=7)

# 左側 colorbar (CE - 下排左): 0.08 - 0.05 - 0.012 = 0.018
cax3 = fig.add_axes([0.018, 0.04, 0.012, 0.44])
cb3 = fig.colorbar(plt.cm.ScalarMappable(cmap='Purples', norm=plt.Normalize(0, 5)), cax=cax3)
cb3.ax.yaxis.set_ticks_position('right')
cb3.ax.yaxis.set_label_position('left')
cb3.set_label('Extreme fire-weather days (%)', fontsize=7, rotation=90, labelpad=5)
cb3.ax.tick_params(labelsize=7)

# 右側 colorbar (CE3 - 下排右): 0.92 + 0.05 = 0.97
cax4 = fig.add_axes([0.97, 0.04, 0.012, 0.44])
cb4 = fig.colorbar(plt.cm.ScalarMappable(cmap='RdPu', norm=plt.Normalize(0, 3)), cax=cax4)
cb4.ax.yaxis.set_ticks_position('left')
cb4.ax.yaxis.set_label_position('right')
cb4.set_label('Preconditional compound\nevents (%)', fontsize=7, rotation=270, labelpad=15)

plt.savefig('Fig_fire_climate_map.png', dpi=300)
print("  已儲存: Fig_fire_climate_map.png")

print("\n" + "=" * 70)
print("  完成!")
print("=" * 70)
