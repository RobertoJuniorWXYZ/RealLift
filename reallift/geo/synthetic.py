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
    random_state=None,
    plot=True,
    verbose=True
) -> dict:
    """
    Run synthetic control analysis.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        treatment_geo (str): Treatment geography.
        control_geos (list): Control geographies.
        treatment_start_date (str): Treatment start date.
        random_state (int or np.random.Generator): Random state for reproducibility.
        plot (bool): Whether to plot.
        verbose (bool): Whether to print results.

    Returns:
        dict: Synthetic control result.
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    treatment_idx = df[df[date_col] >= pd.to_datetime(treatment_start_date)].index[0]

    y = df[treatment_geo].values.astype(float)
    X = df[control_geos].values.astype(float)

    y_mean = y[:treatment_idx].mean()
    X_mean = X[:treatment_idx].mean(axis=0)

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

    post_real = y[treatment_idx:]
    post_synth = synthetic[treatment_idx:]
    effect = post_real - post_synth

    lift_abs = effect.mean()
    effect_pct = effect / post_synth
    lift_pct = np.mean(effect_pct)
    lift_total = effect.sum()

    # Bootstrap
    boot_result = bootstrap_significance(effect, post_synth, random_state=random_state)

    if verbose:
        print("\n=== SYNTHETIC CONTROL RESULT ===")
        print("\nWeights:")
        for geo, w_val in zip(control_geos, weights):
            print(f"{geo}: {w_val:.4f}")
        print(f"\nIntercept: {alpha_val:.4f}")
        print("\nTreatment period:")
        print(f"Start: {df[date_col].iloc[treatment_idx]}")
        print(f"Duration: {len(effect)} days")
        print(f"\nPre-treatment mean error: {pre_error:.2f}")
        print(f"\nMean lift (abs): {lift_abs:.2f}")
        print(f"Mean lift (%): {lift_pct * 100:.2f} %")
        print(f"Total lift: {lift_total:.2f}")

        print("\n=== SIGNIFICANCE ===")
        print("\nT-Test:")
        t_stat, p_value = stats.ttest_1samp(effect, 0)
        print(f"t-stat: {t_stat:.3f}")
        print(f"p-value: {p_value:.4f}")

        print("\n=== BOOTSTRAP TESTS SUMMARY ===")
        print(f"Mean lift (abs): {effect.mean():.2f}")
        print(f"95% CI (abs): [{boot_result['ci_lower_abs']:.2f}, {boot_result['ci_upper_abs']:.2f}]")
        print(f"Mean lift (%): {effect_pct.mean()*100:.2f}%")
        print(f"95% CI (%): [{boot_result['ci_lower_pct']*100:.2f}%, {boot_result['ci_upper_pct']*100:.2f}%]")
        print(f"p-value (bootstrap): {boot_result['p_value_boot']:.4f}")

        if boot_result['ci_lower_abs'] > 0 or boot_result['ci_upper_abs'] < 0:
            print("✔ Lift statistically significant (CI does not cross 0)")
        else:
            print("⚠️ Lift NOT significant (CI crosses 0)")

    if plot:
        plot_synthetic_control(df, treatment_geo, treatment_idx, y, synthetic, effect, post_real, post_synth, effect_pct, boot_result)

    return {
        "weights": dict(zip(control_geos, weights)),
        "alpha": alpha_val,
        "pre_error": pre_error,
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
    
    plt.figure(figsize=(14,5))
    plt.plot(df[date_col], y, label=f"{treatment_geo} Real")
    plt.plot(df[date_col], synthetic, label="Synthetic", linestyle="--")
    plt.axvline(df[date_col].iloc[treatment_idx], linestyle=":", color="black", label="Treatment Start")
    plt.title("GeoLift - Synthetic Control")
    plt.xlabel("Date")
    plt.ylabel("Metric")
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