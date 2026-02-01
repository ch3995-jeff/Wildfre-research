"""
Compound Event Analysis - Final Golden Version
=====================================================
1. Removes Z-score standardization (Preserves physical units).
2. Implements Joint Wald Test (Chi-square) & Paired Test (Direct Bootstrap).
3. Ensures FULL consistency in robust fitting (Jitter fallback everywhere).
4. Includes comprehensive plotting (Fig 1, 1b, 2).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import chi2
from joblib import Parallel, delayed
import os
import warnings
import sys

# Suppress convergence warnings
warnings.filterwarnings('ignore')

# =========================================================================
# Configuration
# =========================================================================
REGION = "WEST_COAST"
N_BOOT = 2000          # 建議維持 2000
N_CORES = -1           # 使用所有核心
RANDOM_SEED = 42

# Threshold 設定
# "1.0" = 只跑 1σ (原本的)
# "all" = 跑所有 threshold (0, 0.5, 1.0, 1.5, 2.0)
THRESHOLD_MODE = "all"  # "1.0" 或 "all"

# Plotting Aesthetics
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

np.random.seed(RANDOM_SEED)

def get_threshold_suffix(threshold):
    """根據 threshold 值返回欄位後綴"""
    if threshold == 1.0:
        return ""  # 原本的欄位沒有後綴
    else:
        return f"_t{int(threshold*10):02d}"

def load_data(filepath, threshold=1.0):
    df = pd.read_csv(filepath)

    # 根據 threshold 選擇欄位
    suffix = get_threshold_suffix(threshold)
    vpd_col = f'VPD_exp_7{suffix}'
    wind_col = f'Wind_exp_7{suffix}'
    ce_col = f'CE_AND_7{suffix}'

    print(f"  使用 Threshold = {threshold}σ (欄位: {vpd_col}, {wind_col}, {ce_col})")

    # 檢查欄位是否存在
    for col in [vpd_col, wind_col, ce_col]:
        if col not in df.columns:
            print(f"  警告: 欄位 {col} 不存在，請先執行 calc_ce_multi_threshold.py")
            return None

    required_cols = ['SPEI12_deficit', 'BurnBndAc', vpd_col, wind_col, ce_col, 'State']
    df = df.dropna(subset=required_cols).copy()

    if REGION == "WEST_COAST":
        states = ['CA', 'OR', 'WA']
    else:
        states = ['CA']

    df = df[df['State'].isin(states)].copy()

    # Variable Definitions - 保持原始物理單位
    df['CE'] = df[ce_col]
    df['VPD'] = df[vpd_col]
    df['Wind'] = df[wind_col]
    df['Dryness'] = df['SPEI12_deficit']
    df['PCE'] = np.sqrt(df['Dryness'] * df['CE'])
    df['log_Burn'] = np.log(df['BurnBndAc'])

    return df

# =========================================================================
# Robust Statistical Functions (Unified)
# =========================================================================

def fit_quantreg_robust(df, formula, tau, return_full=False, jitter_seed=None):
    """
    Robust fitting with increased iterations and Jitter fallback.
    Used by BOTH point estimates and bootstrap loops for consistency.

    Args:
        return_full: If True, return (intercept, slope). If False, return slope only.
        jitter_seed: If provided, use this seed for jitter. Otherwise use RANDOM_SEED + int(tau*1000).
    """
    try:
        model = smf.quantreg(formula, df)
        res = model.fit(q=tau, max_iter=50000, p_tol=1e-5)
        if return_full:
            return res.params.iloc[0], res.params.iloc[1]  # (intercept, slope)
        return res.params.iloc[1]
    except:
        # Fallback: 只 jitter X 欄位 (處理 ties/degeneracy)，不動 Y
        try:
            df_jitter = df.copy()
            x_col = formula.split('~')[1].strip()

            # 用 local RNG，seed 可注入或使用預設
            local_seed = jitter_seed if jitter_seed is not None else (RANDOM_SEED + int(tau * 1000))
            rng = np.random.RandomState(local_seed)
            df_jitter[x_col] += rng.normal(0, 1e-6, size=len(df_jitter))

            model = smf.quantreg(formula, df_jitter)
            res = model.fit(q=tau, max_iter=50000, p_tol=1e-4)
            if return_full:
                return res.params.iloc[0], res.params.iloc[1]
            return res.params.iloc[1]
        except:
            if return_full:
                return np.nan, np.nan
            return np.nan

def joint_wald_test(df, formula, taus_tail, tau_base, n_boot, seed=42):
    """
    Joint Wald Test (Chi-square)
    H0: beta(tau_1) = beta(tau_2) = ... = beta(tau_k) = beta(tau_base)
    """
    # 1. 估計原始係數 (Point Estimates)
    betas_obs = {}
    beta_base = fit_quantreg_robust(df, formula, tau_base)

    for t in taus_tail:
        betas_obs[t] = fit_quantreg_robust(df, formula, t)

    # 檢查是否有點估計失敗
    if np.isnan(beta_base) or any(np.isnan(v) for v in betas_obs.values()):
        return {'Wald_Stat': np.nan, 'P_Value': np.nan, 'DOF': len(taus_tail)}

    theta_obs = np.array([betas_obs[t] - beta_base for t in taus_tail])

    # 2. Bootstrap Covariance
    rng = np.random.RandomState(seed)
    boot_thetas = []

    for i in range(n_boot):
        indices = rng.choice(df.index, size=len(df), replace=True)
        df_boot = df.loc[indices]

        # 必須在同一組樣本上同時算 base 和 tails
        # jitter_seed 綁定 RANDOM_SEED + seed + i + tau
        b_base_boot = fit_quantreg_robust(df_boot, formula, tau_base,
                                          jitter_seed=RANDOM_SEED + seed + i + int(tau_base * 1000))
        row = []
        valid_run = True

        for t in taus_tail:
            b_tail = fit_quantreg_robust(df_boot, formula, t,
                                         jitter_seed=RANDOM_SEED + seed + i + int(t * 1000))
            if np.isnan(b_tail) or np.isnan(b_base_boot):
                valid_run = False
                break
            row.append(b_tail - b_base_boot)

        if valid_run:
            boot_thetas.append(row)

    boot_thetas = np.array(boot_thetas)
    
    # 防呆：如果有效樣本太少
    if len(boot_thetas) < 50:
         return {'Wald_Stat': np.nan, 'P_Value': np.nan, 'DOF': len(taus_tail)}

    # 計算共變異數矩陣
    V_hat = np.cov(boot_thetas, rowvar=False)

    # 3. 計算 Wald Statistic & P-value
    try:
        # 加入微量 regularization 確保可逆
        V_inv = np.linalg.inv(V_hat + np.eye(len(taus_tail)) * 1e-6)
        W_obs = theta_obs.T @ V_inv @ theta_obs

        # P-value (Chi-square)
        p_value_chi2 = 1 - chi2.cdf(W_obs, len(taus_tail))

        return {
            'Wald_Stat': W_obs,
            'P_Value': p_value_chi2,
            'DOF': len(taus_tail),
            'N_Valid_Boot': len(boot_thetas),
            'beta_base': beta_base,
            'betas_tail': betas_obs
        }
    except:
        return {'Wald_Stat': np.nan, 'P_Value': np.nan, 'DOF': len(taus_tail), 'beta_base': beta_base, 'betas_tail': betas_obs}


# =========================================================================
# Analysis Pipeline
# =========================================================================

def analyze_variable(df, var_name, formula, taus, output_dir):
    """
    只跑 Quantile Process (for plotting)，不跑 paired test
    """
    print(f"\nProcessing: {var_name}")

    results_tau = {'tau': [], 'beta': [], 'ci_lo': [], 'ci_hi': []}

    # Bootstrap 次數 (for CI)
    N_BOOT_PLOT = 500

    def process_tau(t):
        beta = fit_quantreg_robust(df, formula, t)

        def get_boot_est(seed):
            rng = np.random.RandomState(seed)
            idx = rng.choice(df.index, size=len(df), replace=True)
            # jitter_seed 綁定 RANDOM_SEED + bootstrap seed + tau
            return fit_quantreg_robust(df.loc[idx], formula, t, jitter_seed=RANDOM_SEED + seed + int(t * 1000))

        boots = [get_boot_est(i) for i in range(N_BOOT_PLOT)]
        boots = [b for b in boots if not np.isnan(b)]

        if not boots: return t, beta, np.nan, np.nan
        return t, beta, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

    print("  Calculating quantile process...")
    results_list = Parallel(n_jobs=N_CORES)(delayed(process_tau)(t) for t in taus)

    for res in results_list:
        results_tau['tau'].append(res[0])
        results_tau['beta'].append(res[1])
        results_tau['ci_lo'].append(res[2])
        results_tau['ci_hi'].append(res[3])

    return pd.DataFrame(results_tau)

# =========================================================================
# Plotting Functions
# =========================================================================

def plot_figure_1_slopes(all_results, output_dir):
    """Figure 1: Coefficient Process Plot (Grid)"""
    print("\nPlotting Figure 1 (Slope Process)...")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    all_vals = []
    for res in all_results.values():
        all_vals.extend(res['ci_hi'])
        all_vals.extend(res['ci_lo'])
    all_vals = [v for v in all_vals if not np.isnan(v)]
    y_min, y_max = min(all_vals)-0.1, max(all_vals)+0.1

    for idx, (name, res) in enumerate(all_results.items()):
        ax = axes[idx]
        ax.fill_between(res['tau'], res['ci_lo'], res['ci_hi'], color='#B22222', alpha=0.2, linewidth=0)
        ax.plot(res['tau'], res['beta'], color='#B22222', linewidth=2)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$\beta(\tau)$')
        ax.set_ylim(y_min, y_max)
        
    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig1_Slope_Process.png", dpi=300)
    plt.close()

def plot_figure_1b_combined(all_results, output_dir):
    """Figure 1b: Combined"""
    print("Plotting Figure 1b (Combined)...")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'VPD':'#E41A1C', 'Wind':'#377EB8', 'Dryness':'#4DAF4A', 'EFW':'#984EA3', 'PCE':'#FF7F00'}
    
    for name, res in all_results.items():
        c = colors.get(name, 'gray')
        ax.plot(res['tau'], res['beta'], color=c, linewidth=2, label=name)
        ax.fill_between(res['tau'], res['ci_lo'], res['ci_hi'], color=c, alpha=0.1)
        
    ax.axhline(0, color='black', linestyle='--')
    ax.legend()
    plt.savefig(f"{output_dir}/Fig1b_Combined.png", dpi=300)
    plt.close()

def plot_figure_2_scatters(df, models, output_dir):
    """Figure 2: Scatter Plots with Quantile Lines"""
    print("Plotting Figure 2 (Scatters)...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # 分位數設定
    plot_taus = [0.15, 0.50, 0.95]
    tau_colors = {0.15: 'black', 0.50: 'red', 0.95: 'red'}
    tau_labels = {0.15: '15th', 0.50: '50th', 0.95: '95th'}

    for idx, (name, formula) in enumerate(models):
        ax = axes[idx]
        x_col = formula.split('~')[1].strip()
        y_col = 'log_Burn'
        x_vals = df[x_col].values
        y_vals = df[y_col].values

        # 散點圖 (用原始資料，與 Figure 1 一致)
        ax.scatter(x_vals, y_vals, s=8, alpha=0.3, color='black', edgecolors='none', zorder=1)

        # 計算 x 軸範圍 (用非零值決定 xlim，提升可讀性)
        x_nonzero = x_vals[x_vals != 0]
        if len(x_nonzero) == 0:
            x_nonzero = x_vals  # fallback
        x_data_min = np.percentile(x_nonzero, 1)
        x_data_max = np.percentile(x_nonzero, 99)
        x_range_span = x_data_max - x_data_min
        x_axis_min = x_data_min - 0.05 * x_range_span
        x_axis_max = x_data_max + 0.25 * x_range_span
        ax.set_xlim(x_axis_min, x_axis_max)

        # 線從軸的 2% 畫到 80% (統一長度)
        x_line_start = x_axis_min + 0.02 * (x_axis_max - x_axis_min)
        x_line_end = x_axis_min + 0.80 * (x_axis_max - x_axis_min)
        x_line = np.linspace(x_line_start, x_line_end, 100)

        for tau in plot_taus:
            # 使用統一的 robust fitting (原始資料，與 Figure 1 一致)
            b0, b1 = fit_quantreg_robust(df, formula, tau, return_full=True)

            if np.isnan(b0) or np.isnan(b1):
                continue

            # 手動計算 y_pred = β0 + β1 * x
            y_pred = b0 + b1 * x_line

            ax.plot(x_line, y_pred, color=tau_colors[tau], linewidth=1.5, zorder=3)

            # 在線的右端標示分位數
            y_end = y_pred[-1]
            ax.text(x_line_end + 0.02 * (x_axis_max - x_axis_min), y_end,
                   tau_labels[tau], fontsize=7, color=tau_colors[tau],
                   va='center', ha='left')

        ax.set_title(name, fontweight='bold')
        ax.set_xlabel(x_col)
        ax.set_ylabel('log(Burned Area)')
        ax.tick_params(direction='in')

    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig2_Scatters.png", dpi=300)
    plt.close()

# =========================================================================
# Main
# =========================================================================

def main(input_csv, output_dir="output_golden"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ACADEMIC ANALYSIS: GOLDEN VERSION")
    print("=" * 80)

    # 決定要跑哪些 threshold
    if THRESHOLD_MODE == "all":
        thresholds = [0.0, 0.5, 1.0, 1.5, 2.0]
        print(f"模式: 跑所有 threshold {thresholds}")
    else:
        thresholds = [float(THRESHOLD_MODE)]
        print(f"模式: 只跑 threshold = {thresholds[0]}σ")

    # 對每個 threshold 執行分析
    for threshold in thresholds:
        print("\n" + "=" * 80)
        print(f"  THRESHOLD = {threshold}σ")
        print("=" * 80)

        df = load_data(input_csv, threshold=threshold)
        if df is None:
            print(f"  跳過 threshold={threshold}σ (欄位不存在)")
            continue

        run_analysis(df, threshold, output_dir)


def run_analysis(df, threshold, output_dir):
    """執行單一 threshold 的分析"""
    # 輸出目錄加上 threshold 後綴 (如果是 all 模式)
    if THRESHOLD_MODE == "all":
        suffix = f"_t{int(threshold*10):02d}"
        this_output_dir = f"{output_dir}{suffix}"
        os.makedirs(this_output_dir, exist_ok=True)
    else:
        this_output_dir = output_dir
    
    models = [
        ('VPD', 'log_Burn ~ VPD'),
        ('Wind', 'log_Burn ~ Wind'),
        ('Dryness', 'log_Burn ~ Dryness'),
        ('EFW', 'log_Burn ~ CE'),
        ('PCE', 'log_Burn ~ PCE'),
    ]
    
    taus_fine = np.arange(0.01, 0.995, 0.01)
    
    all_results = {}

    # 1. Quantile Process for Plotting
    for name, formula in models:
        df_res = analyze_variable(df, name, formula, taus_fine, this_output_dir)
        all_results[name] = df_res

    # 輸出所有分位係數到 CSV
    print("\n  Saving all quantile coefficients...")
    all_coef_rows = []
    for name, df_res in all_results.items():
        for _, row in df_res.iterrows():
            all_coef_rows.append({
                'Variable': name,
                'tau': row['tau'],
                'beta': row['beta'],
                'ci_lo': row['ci_lo'],
                'ci_hi': row['ci_hi']
            })
    df_all_coef = pd.DataFrame(all_coef_rows)
    df_all_coef.to_csv(f"{this_output_dir}/all_quantile_coefficients.csv", index=False)
    print(f"  已儲存: {this_output_dir}/all_quantile_coefficients.csv")

    # =========================================================================
    # Table 1: Quantile Coefficients (展示用)
    # =========================================================================
    print("\n" + "="*80)
    print("TABLE 1: QUANTILE COEFFICIENTS")
    print("="*80)

    taus_display = [0.10, 0.50, 0.90]
    coef_rows = []

    for name, formula in models:
        row = {'Variable': name}
        for t in taus_display:
            beta = fit_quantreg_robust(df, formula, t)
            row[f'Beta_{int(t*100)}'] = beta
        coef_rows.append(row)

    df_coef = pd.DataFrame(coef_rows)

    print("-" * 60)
    print(f"{'Variable':<12} {'β(0.10)':>12} {'β(0.50)':>12} {'β(0.90)':>12}")
    print("-" * 60)
    for _, row in df_coef.iterrows():
        print(f"{row['Variable']:<12} {row['Beta_10']:>12.4f} {row['Beta_50']:>12.4f} {row['Beta_90']:>12.4f}")
    print("-" * 60)

    df_coef.to_csv(f"{this_output_dir}/quantile_coefficients.csv", index=False)
    print(f"  已儲存: {this_output_dir}/quantile_coefficients.csv")

    # =========================================================================
    # Table 2: Joint Wald Test (檢定用)
    # =========================================================================
    print("\n" + "="*80)
    print("TABLE 2: JOINT WALD TEST")
    print("H0: β(0.85) = β(0.90) = β(0.95) = β(0.50)")
    print("="*80)

    def star(p): return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else ''

    joint_results = []
    taus_tail = [0.85, 0.90, 0.95]
    tau_base = 0.50

    for name, formula in models:
        print(f"  Testing {name}...")
        res = joint_wald_test(df, formula, taus_tail, tau_base, N_BOOT)
        res['Variable'] = name
        joint_results.append(res)

    print("-" * 70)
    print(f"{'Variable':<12} {'Wald χ²':>12} {'DOF':>8} {'P_Value':>12} {'Sig':>8}")
    print("-" * 70)

    test_rows = []
    for res in joint_results:
        var = res['Variable']
        p = res['P_Value']
        sig = star(p)
        print(f"{var:<12} {res['Wald_Stat']:>12.4f} {res['DOF']:>8} {p:>12.4f} {sig:>8}")

        test_rows.append({
            'Variable': var,
            'Wald_Chi2': res['Wald_Stat'],
            'DOF': res['DOF'],
            'P_Value': p,
            'Sig': sig
        })

    print("-" * 70)

    df_test = pd.DataFrame(test_rows)
    df_test.to_csv(f"{this_output_dir}/joint_wald_test.csv", index=False)
    print(f"  已儲存: {this_output_dir}/joint_wald_test.csv")

    # Plotting
    plot_figure_1_slopes(all_results, this_output_dir)
    plot_figure_1b_combined(all_results, this_output_dir)
    plot_figure_2_scatters(df, models, this_output_dir)
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data_ce_continuous.csv"
    main(input_file)