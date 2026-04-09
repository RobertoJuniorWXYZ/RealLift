import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy import stats
from ..geo.bootstrap import bootstrap_significance

def run_synthetic_control(
    filepath,
    date_col,
    treatment_geo,
    control_geos,
    treatment_start_date,
    treatment_end_date=None,
    start_date=None,
    end_date=None,
    random_state=None,
    cluster_idx=None,
    plot=True,
    verbose=True
) -> dict:
    """
    Run synthetic control analysis.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        treatment_geo (str or list): Treatment geography or list of geographies.
        control_geos (list): Control geographies.
        treatment_start_date (str): Treatment start date, separating pre-test and post-test periods.
        treatment_end_date (str): Treatment end date (analysis window end).
        start_date (str): YYYY-MM-DD date when analysis begins (optional).
        end_date (str): YYYY-MM-DD date when analysis ends (optional).
        random_state (int or np.random.Generator): Random state for reproducible permutations.
        cluster_idx (int or str): Optional cluster index for logging traceability.
        plot (bool): Whether to plot the baseline graph.
        verbose (bool): Whether to print verbose logging results.

    Returns:
        dict: Synthetic control result.
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col])

    # Period Filtering (Global Window)
    if start_date is not None:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.to_datetime(end_date)]

    df = df.sort_values(date_col).reset_index(drop=True)

    # 1. Determine causal split index (Pre vs Post)
    post_data = df[df[date_col] >= pd.to_datetime(treatment_start_date)]
    is_pre_only = len(post_data) == 0

    if is_pre_only:
        treatment_idx = len(df)
        treatment_end_idx = len(df)
    else:
        treatment_idx = post_data.index[0]
        # Determine treatment end index for EFFECT calculation
        if treatment_end_date:
            end_mask = df[date_col] <= pd.to_datetime(treatment_end_date)
            treatment_end_idx = df[end_mask].index[-1] + 1 if any(end_mask) else len(df)
        else:
            treatment_end_idx = len(df)

    if isinstance(treatment_geo, list):
        y = df[treatment_geo].mean(axis=1).values.astype(float)
    else:
        y = df[treatment_geo].values.astype(float)

    X = df[control_geos].values.astype(float)

    y_mean = y[:treatment_idx].mean() if treatment_idx > 0 else 1e-10
    if y_mean == 0: y_mean = 1e-10
    X_mean = X[:treatment_idx].mean(axis=0) if treatment_idx > 0 else X.mean(axis=0) + 1e-10
    X_mean[X_mean == 0] = 1e-10

    y_norm = y / y_mean
    X_norm = X / X_mean

    X_pre = np.nan_to_num(X_norm[:treatment_idx])
    y_pre = np.nan_to_num(y_norm[:treatment_idx])

    w = cp.Variable(len(control_geos))
    alpha = cp.Variable()

    objective = cp.Minimize(cp.sum_squares(y_pre - (X_pre @ w + alpha)))
    constraints = [w >= 0, cp.sum(w) == 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Solver failed: {problem.status}")

    weights = np.array(w.value).flatten()
    alpha_val = float(alpha.value)

    synthetic_norm = X_norm @ weights + alpha_val
    synthetic = synthetic_norm * y_mean

    df["synthetic"] = synthetic

    pre_real = y[:treatment_idx]
    pre_synth = synthetic[:treatment_idx]
    pre_error = np.mean(np.abs(pre_real - pre_synth))
    pre_mspe = np.mean((pre_real - pre_synth)**2)

    if is_pre_only:
        post_real = np.array([])
        post_synth = np.array([])
        effect = np.array([])
        lift_abs = 0.0
        effect_pct = np.array([])
        lift_pct = 0.0
        lift_total = 0.0
        boot_result = {
            "p_value": 1.0,
            "ci_lower_pct": 0.0, "ci_upper_pct": 0.0,
            "ci_lower_total_pct": 0.0, "ci_upper_total_pct": 0.0,
            "ci_lower_abs": 0.0, "ci_upper_abs": 0.0,
            "ci_lower_total_abs": 0.0, "ci_upper_total_abs": 0.0
        }
        post_mspe = 0.0
    else:
        post_real = y[treatment_idx : treatment_end_idx]
        post_synth = synthetic[treatment_idx : treatment_end_idx]
        effect = post_real - post_synth

        lift_abs = effect.mean()
        effect_pct = effect / post_synth if post_synth.sum() != 0 else np.zeros_like(effect)
        lift_pct = np.mean(effect_pct)
        lift_total = effect.sum()
        post_mspe = np.mean(effect**2)

        # Bootstrap
        boot_result = bootstrap_significance(effect, post_synth, random_state=random_state)

    if verbose:
        header = "=== GEO SYNTHETIC CONTROL ==="
        if cluster_idx is not None:
            header = f"=== GEO SYNTHETIC CONTROL (Cluster {cluster_idx}) ==="
        print(f"\n{header}")
        print("\nWeights:")
        for geo, w_val in zip(control_geos, weights):
            if w_val > 0.001:
                print(f"{geo}: {w_val:.4f}")

        print("\nTreatment period:")
        start_date_str = df[date_col].iloc[treatment_idx].strftime('%Y-%m-%d')
        end_date_str = df[date_col].iloc[treatment_end_idx - 1].strftime('%Y-%m-%d')
        print(f"Start: {start_date_str}")
        print(f"End: {end_date_str}")
        print(f"Duration: {len(effect)} days")

        print(f"\nMean lift (abs): {lift_abs:.2f}")
        print(f"Mean lift (%): {lift_pct * 100:.2f} %")
        print(f"Total lift: {lift_total:.2f}")

        sig_header = "=== SIGNIFICANCE ==="
        boot_header = "=== BOOTSTRAP TESTS SUMMARY ==="
        if cluster_idx is not None:
            sig_header = f"=== SIGNIFICANCE (Cluster {cluster_idx}) ==="
            boot_header = f"=== BOOTSTRAP TESTS SUMMARY (Cluster {cluster_idx}) ==="

        print(f"\n{sig_header}")
        print("\nT-Test:")
        t_stat, p_value = stats.ttest_1samp(effect, 0)
        print(f"t-stat: {t_stat:.3f}")
        print(f"p-value: {p_value:.4f}")

        print(f"\n{boot_header}")
        print(f"Total lift (abs): {lift_total:.2f}")
        print(f"95% CI (abs): [{boot_result['ci_lower_total_abs']:.2f}, {boot_result['ci_upper_total_abs']:.2f}]")
        
        total_lift_pct = lift_total / post_synth.sum()
        print(f"Total lift (%): {total_lift_pct*100:.2f}%")
        print(f"95% CI (%): [{boot_result['ci_lower_total_pct']*100:.2f}%, {boot_result['ci_upper_total_pct']*100:.2f}%]")
        print(f"p-value (bootstrap): {boot_result['p_value_boot']:.4f}")

        if boot_result['ci_lower_total_abs'] > 0 or boot_result['ci_upper_total_abs'] < 0:
            print("[Yes] Lift statistically significant (CI does not cross 0)")
        else:
            print("[No] Lift NOT significant (CI crosses 0)")

    if plot:
        plot_synthetic_control(df, treatment_geo, treatment_idx, y, synthetic, effect, post_real, post_synth, effect_pct, boot_result)

    return {
        "weights": dict(zip(control_geos, weights)),
        "alpha": alpha_val,
        "pre_error": pre_error,
        "pre_mspe": pre_mspe,
        "post_mspe": post_mspe,
        "mspe_ratio": post_mspe / pre_mspe if pre_mspe > 0 else 0,
        "lift_mean_abs": lift_abs,
        "lift_mean_pct": lift_pct,
        "lift_total": lift_total,
        "df": df,
        "bootstrap": boot_result,
        "plotting_data": {
            "treatment_geo": treatment_geo,
            "treatment_idx": treatment_idx,
            "y": y,
            "synthetic": synthetic,
            "effect": effect,
            "post_real": post_real,
            "post_synth": post_synth,
            "effect_pct": effect_pct
        }
    }

def plot_synthetic_control(df, treatment_geo, treatment_idx, y, synthetic, effect, post_real, post_synth, effect_pct, boot_result):
    """
    Generate plots for synthetic control analysis.
    """
    date_col = df.columns[0] # Assuming first col is date as per common usage
    
    geo_name = ", ".join(treatment_geo) if isinstance(treatment_geo, list) else treatment_geo

    plt.figure(figsize=(14,5))
    plt.plot(df[date_col], y, label=f"{geo_name} Real")
    plt.plot(df[date_col], synthetic, label="Synthetic", linestyle="--")
    plt.axvline(df[date_col].iloc[treatment_idx], linestyle=":", color="black", label="Treatment Start")
    plt.title(f"GeoLift - Synthetic Control ({geo_name})")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14,4))
    plt.plot(df[date_col].iloc[treatment_idx:], effect, label="Daily lift")
    plt.axhline(0, linestyle="--")
    plt.title("Lift (Treatment - Synthetic)")
    plt.xlabel("Date")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    baseline = post_synth.mean()
    observed = post_real.mean()
    plt.figure(figsize=(6,5))
    plt.bar(["Synthetic", "Real"], [baseline, observed])
    plt.title("Average Outcome (Post Period)")
    plt.ylabel("Value")
    plt.show()

    plt.figure(figsize=(8,5))
    plt.hist(boot_result['boot_means_pct'] * 100, bins=40)
    plt.axvline(effect_pct.mean() * 100, linestyle="--", label="Observed lift (%)")
    plt.axvline(0, linestyle="--", label="Zero")
    plt.title("Lift Distribution (Bootstrap - %)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
