import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from ..config.defaults import DEFAULT_POWER, DEFAULT_ALPHA_TEST, DEFAULT_MAX_DAYS, DEFAULT_MDE

def estimate_duration(
    filepath,
    date_col,
    treatment_geo,
    control_geos,
    mde=DEFAULT_MDE,
    alpha=DEFAULT_ALPHA_TEST,
    power_target=DEFAULT_POWER,
    max_days=DEFAULT_MAX_DAYS,
    treatment_start_date=None,
    cluster_idx=None,
    verbose=True
) -> dict:
    """
    Estimate the duration needed for a GeoLift experiment.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        treatment_geo (str): Treatment geography.
        control_geos (list): Control geographies.
        mde (float): Minimum detectable effect.
        alpha (float): Significance level.
        power_target (float): Target power (typically 0.8 / 80%).
        max_days (int or list): Maximum days limit for the duration, or a range [min, max] like [21, 60].
        treatment_start_date (str): Treatment start date to filter data exclusively to the pre-treatment period.
        cluster_idx (int or str): Optional cluster index for logging traceability.
        verbose (bool): Whether to print logging results.

    Returns:
        dict: Duration estimation result.
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col])
    
    if treatment_start_date is not None:
        df = df[df[date_col] < pd.to_datetime(treatment_start_date)]

    df = df.sort_values(date_col)

    start_date_str = df[date_col].iloc[0].strftime('%Y-%m-%d')
    end_date_str = df[date_col].iloc[-1].strftime('%Y-%m-%d')

    if isinstance(treatment_geo, list):
        for g in treatment_geo:
            if g not in df.columns:
                raise ValueError(f"Treatment geo {g} not found")
        treatment = df[treatment_geo].mean(axis=1)
    else:
        if treatment_geo not in df.columns:
            raise ValueError(f"Treatment geo {treatment_geo} not found")
        treatment = df[treatment_geo]

    for g in control_geos:
        if g not in df.columns:
            raise ValueError(f"{g} not found")

    controls_df = df[control_geos]

    mean_treat = treatment.mean()
    std_naive = treatment.std()

    control_mean = controls_df.mean(axis=1)
    corr = treatment.corr(control_mean)

    residual_simple = treatment - control_mean
    std_residual_simple = residual_simple.std()

    if isinstance(treatment_geo, list):
        # Use log-diff on the mean treatment series
        df_log_diff = np.log(pd.concat([treatment, df[control_geos]], axis=1)).diff().dropna()
        y = df_log_diff.iloc[:, 0].values
        X = df_log_diff.iloc[:, 1:].values
    else:
        df_transformed = np.log(df[[treatment_geo] + control_geos]).diff().dropna()
        y = df_transformed[treatment_geo].values
        X = df_transformed[control_geos].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    residual_reg = y - y_pred
    std_residual_reg = residual_reg.std()

    delta = np.log(1 + mde)
    delta_pct = np.exp(delta) - 1
    delta_abs = mean_treat * delta_pct

    def compute_power(effect, std, n, alpha):
        se = std / np.sqrt(n)
        z = effect / se
        z_alpha = norm.ppf(1 - alpha/2)
        return norm.cdf(z - z_alpha)

    if isinstance(max_days, int):
        days_list = list(range(7, max_days + 1))
    elif isinstance(max_days, (list, tuple)) and len(max_days) == 2:
        days_list = list(range(max_days[0], max_days[1] + 1))
    else:
        days_list = sorted(max_days)

    results = []
    for d in days_list:
        results.append({
            "days": d,
            "power_naive": compute_power(delta, std_naive, d, alpha),
            "power_residual_simple": compute_power(delta, std_residual_simple, d, alpha),
            "power_regression": compute_power(delta, std_residual_reg, d, alpha)
        })

    results_df = pd.DataFrame(results)

    valid = results_df[results_df["power_regression"] >= power_target]

    if not valid.empty:
        best = valid.iloc[0]
        best_days = int(best["days"])
        best_power = best["power_regression"]
        estimated_days = best_days
    else:
        best_days = None
        best_power = None
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power_target)
        n_est = ((z_alpha + z_beta) * std_residual_reg / delta) ** 2
        estimated_days = int(np.ceil(n_est))

    if verbose:
        header = "=== GEO DURATION ESTIMATION ==="
        if cluster_idx is not None:
            header = f"=== GEO DURATION ESTIMATION (Cluster {cluster_idx}) ==="
        print(f"\n{header}")
        print(f"Treatment: {treatment_geo}")
        print(f"Control: {control_geos}")
        
        print("\n=== EVALUATING PERIOD ===")
        print(f"Start Date: {start_date_str}")
        print(f"End Date: {end_date_str}")

        print("\n=== REGRESSION STATISTICS ===")
        print(f"Mean: {mean_treat:.2f}")
        print(f"Std Residual: {std_residual_reg:.4f}")
        print(f"R-Squared: {model.score(X, y):.4f}")
        print(f"Correlation: {corr:.3f}")
        print("\n=== MDE ===")
        print(f"MDE: {mde*100:.2f}%")
        print(f"Effect absolute: {delta_abs:.2f}")
        print(f"Effect percent real: {delta_pct*100:.2f}%")
        print("\n=== RESULT ===")
        if best_days:
            print(f"✔ Min duration: {best_days} days")
            print(f"✔ Power: {best_power:.2%}")
        else:
            print("⚠️ Did not reach target power in tested days")
            print(f"👉 Estimated days needed: {estimated_days} days")

    return {
        "summary": {
            "start_date": start_date_str,
            "end_date": end_date_str,
            "mean": mean_treat,
            "std_naive": std_naive,
            "std_residual_simple": std_residual_simple,
            "std_residual_regression": std_residual_reg,
            "r_squared": model.score(X, y),
            "correlation": corr,
            "mde": mde,
            "delta_log": delta,
            "delta_pct": delta_pct,
            "delta_abs": delta_abs,
            "best_days": best_days,
            "best_power": best_power,
            "estimated_days_needed": estimated_days
        },
        "power_curve": results_df
    }