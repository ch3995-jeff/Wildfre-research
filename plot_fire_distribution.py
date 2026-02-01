"""
plot_fire_distribution.py
火災面積分布圖 - Median + 95th percentile 版本
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 載入資料
df = pd.read_csv('data_ce_continuous.csv')
df = df[df['State'].isin(['CA', 'OR', 'WA'])].copy()
df = df.dropna(subset=['BurnBndAc'])
burn_area = df['BurnBndAc'].values

# 用 log scale
log_burn = np.log10(burn_area[burn_area > 0])

# 統計量
median_log = np.percentile(log_burn, 50)
p90_log = np.percentile(log_burn, 90)

print(f"Median (50th): {median_log:.2f} = {10**median_log:,.0f} acres")
print(f"90th percentile: {p90_log:.2f} = {10**p90_log:,.0f} acres")
print(f"N fires: {len(log_burn):,}")

# KDE
kde = gaussian_kde(log_burn, bw_method=0.3)
x_log = np.linspace(log_burn.min(), log_burn.max(), 500)
y_kde = kde(x_log)

# 繪圖
fig, ax = plt.subplots(figsize=(8, 5))

# KDE 填色
ax.fill_between(x_log, y_kde, alpha=0.3, color='steelblue')
ax.plot(x_log, y_kde, color='steelblue', linewidth=2)

# Median 虛線
ax.axvline(median_log, color='#333333', linestyle='--', linewidth=1.5)
ax.text(median_log + 0.05, y_kde.max() * 0.85, f'Median\n{10**median_log:,.0f} ac',
        fontsize=9, ha='left', va='top')

# 90th percentile 虛線
ax.axvline(p90_log, color='darkred', linestyle='--', linewidth=1.5)
ax.text(p90_log + 0.05, y_kde.max() * 0.6, f'90th\n{10**p90_log:,.0f} ac',
        fontsize=9, ha='left', va='top', color='darkred')

# 軸設定
ax.set_xlabel('Fire Size (acres)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Fire Size Distribution (CA, OR, WA)', fontsize=12, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# X 軸刻度
ax.set_xticks([2, 3, 4, 5, 6])
ax.set_xticklabels(['100', '1K', '10K', '100K', '1M'])

# 緊湊範圍
ax.set_ylim(0, y_kde.max() * 1.05)
ax.set_xlim(log_burn.min(), log_burn.max())

plt.tight_layout()
plt.savefig('output_golden/Fig_fire_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n已儲存: output_golden/Fig_fire_distribution.png")
