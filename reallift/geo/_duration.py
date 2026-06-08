import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from ..config.defaults import DEFAULT_POWER, DEFAULT_ALPHA_TEST, DEFAULT_EXPERIMENT_DAYS, DEFAULT_MDE


def _simulate_mde(residuals_logdiff, mean_treat, d, alpha, power_target,
                  fixed_mde=None, n_sim=300, n_boot=500, n_delta=50):
    """
    MDE estimation via lift simulation + MBB.
    Fully consistent with the inference method: injects a hypothetical lift
    into MBB-resampled noise and checks empirical rejection rate.

    Uses a vectorized trick: precomputes the bootstrap CI lower bounds for
    the null (delta=0), then shifts by delta_abs — O(n_sim) per delta value.

    Args:
        residuals_logdiff: calibration regression residuals in log-diff space.
        mean_treat: pre-period mean of treatment geo in level space.
        d: experiment duration in days.
        alpha: significance level (e.g. 0.05).
        power_target: target power (e.g. 0.80).
        fixed_mde: if provided, return power at this specific lift (fraction).
        n_sim: number of outer noise simulations.
        n_boot: number of inner bootstrap iterations per simulation.
        n_delta: number of delta values in the search grid.

    Returns:
        (mde_pct, mde_power) when fixed_mde is None.
        float power when fixed_mde is provided.
    """
    resid_level = np.array(residuals_logdiff) * mean_treat
    n_resid = len(resid_level)

    if n_resid >= d:
        resid_d = resid_level[-d:]
    else:
        rng0 = np.random.default_rng(0)
        resid_d = resid_level[rng0.choice(n_resid, size=d, replace=True)]

    n = len(resid_d)
    block_size = 7 if n >= 14 else (3 if n >= 6 else 1)
    ci_lower_p = (alpha / 2.0) * 100
    rng = np.random.default_rng(42)

    def _mbb_rows(n_rows):
        """Generate n_rows MBB resamples of resid_d. Returns (n_rows, n)."""
        mat = np.zeros((n_rows, n))
        if block_size > 1:
            pool = np.arange(n - block_size + 1)
            nb = int(np.ceil(n / block_size))
            for i in range(n_rows):
                starts = rng.choice(pool, size=nb, replace=True)
                idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
                mat[i] = resid_d[idx]
        else:
            for i in range(n_rows):
                mat[i] = resid_d[rng.choice(n, size=n, replace=True)]
        return mat

    # Outer: n_sim noise realizations (level-space MBB resamples of residuals)
    noise_matrix = _mbb_rows(n_sim)   # (n_sim, n)

    # Inner: n_boot bootstrap index matrix
    if block_size > 1:
        pool = np.arange(n - block_size + 1)
        nb = int(np.ceil(n / block_size))
        boot_idx = np.zeros((n_boot, n), dtype=int)
        for b in range(n_boot):
            starts = rng.choice(pool, size=nb, replace=True)
            idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
            boot_idx[b] = idx
    else:
        boot_idx = rng.integers(0, n, size=(n_boot, n))

    # Precompute bootstrap CI lower bounds under null (delta=0).
    # noise_matrix[:, boot_idx] → (n_sim, n_boot, n); mean over axis=2 → (n_sim, n_boot)
    boot_means_null = noise_matrix[:, boot_idx].mean(axis=2)           # (n_sim, n_boot)
    ci_lowers_null  = np.percentile(boot_means_null, ci_lower_p, axis=1)  # (n_sim,)

    # KEY: adding delta_abs shifts every CI lower by +delta_abs.
    # Rejection ↔ ci_lower_null + delta_abs > 0 ↔ ci_lower_null > -delta_abs
    def _power_at(delta_pct):
        return float(np.mean(ci_lowers_null > -(delta_pct * mean_treat)))

    if fixed_mde is not None:
        return _power_at(fixed_mde)

    delta_grid = np.linspace(0.001, 0.30, n_delta)
    powers     = np.array([_power_at(d_pct) for d_pct in delta_grid])

    reached  = np.where(powers >= power_target)[0]
    mde_pct  = float(delta_grid[reached[0]])  if len(reached) > 0 else float(delta_grid[-1])
    mde_power = float(powers[reached[0]])      if len(reached) > 0 else float(powers[-1])

    return mde_pct, mde_power

def estimate_duration(
    filepath=None,
    date_col=None,
    treatment_geo=None,
    control_geos=None,
    control_weights=None,
    clusters=None,
    mde=DEFAULT_MDE,
    alpha=DEFAULT_ALPHA_TEST,
    power_target=DEFAULT_POWER,
    experiment_days=DEFAULT_EXPERIMENT_DAYS,
    start_date=None,
    end_date=None,
    cluster_idx=None,
    consolidated=False,
    cluster_residuals=None,
    verbose=True,
    use_bootstrap_mde=True,
    df=None
) -> dict:
    """
    Estimate the duration needed for a GeoLift experiment.

    When mde is provided (default), the function computes a power curve across
    candidate days and finds the minimum duration that achieves the target power.

    When mde=None, the function inverts the analysis: for each candidate number
    of days, it computes the Minimum Detectable Effect (MDE) achievable at the
    target power level.

    When clusters is provided (output of discover_geo_clusters), the function
    automatically runs per-cluster analysis followed by a consolidated MDE
    using the averaged regression residuals from all clusters.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        treatment_geo (str or list): Treatment geography or list of treatment geos.
        control_geos (list or None): Control geographies. Required for per-cluster mode.
        clusters (list of dict): Output from discover_geo_clusters. Each dict has
            'treatment' (list) and 'control' (list). When provided, runs per-cluster
            analysis + consolidated automatically.
        mde (float or None): Minimum detectable effect. If None, MDE is computed
            automatically for each candidate duration.
        alpha (float): Significance level.
        power_target (float): Target power (typically 0.8 / 80%).
        experiment_days (int or list): Experiment duration range, e.g. [21, 60].
        start_date (str): YYYY-MM-DD date when history begins (optional).
        end_date (str): YYYY-MM-DD date when history ends (optional).
        cluster_idx (int or str): Optional cluster index for logging traceability.
        consolidated (bool): If True, computes experiment-level MDE using per-cluster
            regression residuals (via cluster_residuals) or treatment-only variance.
        cluster_residuals (list of Series): Residual time series from per-cluster regressions.
        verbose (bool): Whether to print logging results.
        df (pd.DataFrame, optional): Pre-loaded DataFrame. When provided, skips CSV I/O.

    Returns:
        dict: Duration estimation result.
    """
    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-CLUSTER MODE: auto-orchestrate per-cluster + consolidated
    # ═══════════════════════════════════════════════════════════════════════
    if clusters is not None:
        cluster_results = []
        all_residuals = []

        for i, cluster in enumerate(clusters):
            treat = cluster["treatment"][0] if len(cluster["treatment"]) == 1 else cluster["treatment"]
            res = estimate_duration(
                filepath=filepath,
                date_col=date_col,
                treatment_geo=treat,
                control_geos=cluster["control"].copy(),
                control_weights=cluster.get("control_weights", []).copy(),
                mde=mde,
                alpha=alpha,
                power_target=power_target,
                experiment_days=experiment_days,
                start_date=start_date,
                end_date=end_date,
                cluster_idx=i,
                verbose=verbose,
                use_bootstrap_mde=use_bootstrap_mde,
                df=df
            )
            cluster_results.append(res)
            if res.get("residuals") is not None:
                all_residuals.append(res["residuals"])

        # Consolidated
        all_treatments = [c["treatment"][0] for c in clusters]
        consolidated_res = estimate_duration(
            filepath=filepath,
            date_col=date_col,
            treatment_geo=all_treatments,
            mde=mde,
            alpha=alpha,
            power_target=power_target,
            experiment_days=experiment_days,
            start_date=start_date,
            end_date=end_date,
            consolidated=True,
            cluster_residuals=all_residuals if all_residuals else None,
            verbose=verbose,
            df=df
        )

        return {
            "cluster_results": cluster_results,
            "consolidated": consolidated_res,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE CALL MODE (per-cluster or consolidated)
    # ═══════════════════════════════════════════════════════════════════════
    if treatment_geo is None:
        raise ValueError("Either 'clusters' or 'treatment_geo' must be provided.")
    if df is not None:
        df = df.copy()
    else:
        if filepath is None:
            raise ValueError("Either 'filepath' or 'df' must be provided.")
        df = pd.read_csv(filepath)
        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
        df = df.dropna(subset=[date_col])
    
    if start_date is not None:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.to_datetime(end_date)]

    df = df.sort_values(date_col)

    start_date_str = df[date_col].iloc[0].strftime('%Y-%m-%d')
    end_date_str = df[date_col].iloc[-1].strftime('%Y-%m-%d')

    # Validate treatment geos
    if isinstance(treatment_geo, list):
        for g in treatment_geo:
            if g not in df.columns:
                raise ValueError(f"Treatment geo {g} not found")
        treatment = df[treatment_geo].mean(axis=1)
    else:
        if treatment_geo not in df.columns:
            raise ValueError(f"Treatment geo {treatment_geo} not found")
        treatment = df[treatment_geo]

    mean_treat = treatment.mean()

    # Build the list of candidate days
    if isinstance(experiment_days, int):
        days_list = list(range(7, experiment_days + 1))
    elif isinstance(experiment_days, (list, tuple)) and len(experiment_days) == 2:
        days_list = list(range(experiment_days[0], experiment_days[1] + 1))
    else:
        days_list = sorted(experiment_days)

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power_target)

    # ═══════════════════════════════════════════════════════════════════════
    # CONSOLIDATED MODE: variance of per-cluster residuals (or treatment)
    # ═══════════════════════════════════════════════════════════════════════
    if consolidated:
        if cluster_residuals is not None and len(cluster_residuals) > 0:
            # Average the per-cluster residual series → variance of the mean
            residuals_df = pd.concat(
                [r.reset_index(drop=True) for r in cluster_residuals],
                axis=1
            )
            mean_residuals = residuals_df.mean(axis=1)
            sigma = mean_residuals.std()
            method = "residuals"
            n_series = len(cluster_residuals)
        else:
            # Fallback: variance of the aggregated treatment log-diffs
            if isinstance(treatment_geo, list) and len(treatment_geo) > 1:
                log_diffs = np.log(df[treatment_geo]).diff().dropna()
                treatment_mean_logdiff = log_diffs.mean(axis=1)
            elif isinstance(treatment_geo, list):
                treatment_mean_logdiff = np.log(df[treatment_geo[0]]).diff().dropna()
            else:
                treatment_mean_logdiff = np.log(df[treatment_geo]).diff().dropna()
            sigma = treatment_mean_logdiff.std()
            method = "treatment_variance"
            n_series = len(treatment_geo) if isinstance(treatment_geo, list) else 1

        return _compute_and_report(
            mde=mde,
            alpha=alpha,
            power_target=power_target,
            days_list=days_list,
            z_alpha=z_alpha,
            z_beta=z_beta,
            sigma=sigma,
            mean_treat=mean_treat,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            treatment_geo=treatment_geo,
            control_geos=None,
            cluster_idx=cluster_idx,
            consolidated=True,
            verbose=verbose,
            use_bootstrap_mde=False,
            extra_stats={
                "method": method,
                "n_series": n_series,
            }
        )

    # ── PER-CLUSTER MODE: regression with optimized controls ────────────────
    if control_geos is None:
        raise ValueError("control_geos is required when consolidated=False")

    for g in control_geos:
        if g not in df.columns:
            raise ValueError(f"{g} not found")

    controls_df = df[control_geos]
    std_naive = treatment.std()

    # Standard diagnostics (Simple Control Mean)
    control_mean = controls_df.mean(axis=1)
    corr = treatment.corr(control_mean)
    std_residual_simple = (treatment - control_mean).std()

    df_transformed = np.log(pd.concat([treatment, df[control_geos]], axis=1)).diff().dropna()
    y = df_transformed.iloc[:, 0].values
    X = df_transformed.iloc[:, 1:].values

    if control_weights is not None and len(control_weights) == X.shape[1]:
        # Bivariate calibration regression on the synthetic composite:
        # Δlog Y = α + β·Δlog Ŷ_synthetic + ε  (preprint Section 4.7)
        y_synthetic = (X @ np.array(control_weights)).reshape(-1, 1)
        model = LinearRegression().fit(y_synthetic, y)
        y_pred = model.predict(y_synthetic)
        r_squared = model.score(y_synthetic, y)
    else:
        # Fallback: multivariate OLS on all controls (no SCM weights available)
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r_squared = model.score(X, y)

    residual_reg = y - y_pred
    std_residual_reg = residual_reg.std()

    return _compute_and_report(
        mde=mde,
        alpha=alpha,
        power_target=power_target,
        days_list=days_list,
        z_alpha=z_alpha,
        z_beta=z_beta,
        sigma=std_residual_reg,
        mean_treat=mean_treat,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        treatment_geo=treatment_geo,
        control_geos=control_geos,
        cluster_idx=cluster_idx,
        consolidated=False,
        verbose=verbose,
        use_bootstrap_mde=use_bootstrap_mde,
        extra_stats={
            "std_naive": std_naive,
            "std_residual_simple": std_residual_simple,
            "std_residual_regression": std_residual_reg,
            "r_squared": r_squared,
            "correlation": corr,
        },
        residuals=pd.Series(residual_reg)
    )


def _compute_and_report(
    mde, alpha, power_target,
    days_list, z_alpha, z_beta, sigma,
    mean_treat, start_date_str, end_date_str,
    treatment_geo, control_geos,
    cluster_idx, consolidated, verbose,
    extra_stats, residuals=None,
    use_bootstrap_mde=True
):
    """
    Internal: compute MDE curve or power curve and produce output.
    Shared between consolidated and per-cluster modes.
    """
    # ── AUTO-MDE MODE (mde=None) ─────────────────────────────────────────
    if mde is None:
        mde_curve = []
        for d in days_list:
            if use_bootstrap_mde and residuals is not None:
                mde_pct, _ = _simulate_mde(residuals, mean_treat, d, alpha, power_target)
                mde_curve.append({"days": d, "mde": mde_pct})
            else:
                se = sigma / np.sqrt(max(d, 1))
                delta = (z_alpha + z_beta) * se
                mde_curve.append({"days": d, "mde": np.exp(delta) - 1})

        mde_df = pd.DataFrame(mde_curve)

        if verbose:
            _print_header(consolidated, cluster_idx, auto_mde=True)
            _print_common_info(treatment_geo, control_geos, start_date_str, end_date_str)
            _print_stats(consolidated, sigma, mean_treat, extra_stats)

            print(f"\n=== MDE CURVE (power={power_target:.0%}, α={alpha}) ===")
            print(f"{'Days':<6} | {'MDE':<16}")
            print("-" * 26)
            display_days = [d for d in days_list if d % 7 == 0 or d == days_list[0] or d == days_list[-1]]
            for _, row in mde_df[mde_df["days"].isin(display_days)].iterrows():
                print(f"{int(row['days']):<6} | {row['mde']*100:<15.2f}%")

        summary = _build_summary(
            start_date_str, end_date_str, mean_treat, sigma,
            consolidated, extra_stats,
            mde=None, delta_log=None, delta_pct=None, delta_abs=None,
            best_days=None, best_power=None, estimated_days_needed=None,
            auto_mde=True
        )
        return {"summary": summary, "mde_curve": mde_df, "residuals": residuals}

    # ── FIXED-MDE MODE ───────────────────────────────────────────────────
    delta = np.log(1 + mde)
    delta_pct = np.exp(delta) - 1
    delta_abs = mean_treat * delta_pct

    def compute_power(effect, se, a):
        z = effect / se
        z_a = norm.ppf(1 - a / 2)
        return norm.cdf(z - z_a)

    results = []
    for d in days_list:
        if use_bootstrap_mde and residuals is not None:
            power = _simulate_mde(residuals, mean_treat, d, alpha, power_target, fixed_mde=mde)
        else:
            se = sigma / np.sqrt(max(d, 1))
            power = compute_power(delta, se, alpha)
        results.append({"days": d, "power": power})

    results_df = pd.DataFrame(results)
    valid = results_df[results_df["power"] >= power_target]

    if not valid.empty:
        best = valid.iloc[0]
        best_days = int(best["days"])
        best_power = best["power"]
        estimated_days = best_days
    else:
        best_days = None
        best_power = None
        estimated_days = int(np.ceil(((z_alpha + z_beta) * sigma / delta) ** 2))

    if verbose:
        _print_header(consolidated, cluster_idx, auto_mde=False)
        _print_common_info(treatment_geo, control_geos, start_date_str, end_date_str)
        _print_stats(consolidated, sigma, mean_treat, extra_stats)

        print("\n=== MDE ===")
        print(f"MDE: {mde*100:.2f}%")
        print(f"Effect absolute: {delta_abs:.2f}")
        print(f"Effect percent real: {delta_pct*100:.2f}%")
        print("\n=== RESULT ===")
        if best_days:
            print(f"[Yes] Min duration: {best_days} days")
            print(f"[Yes] Power: {best_power:.2%}")
        else:
            print("[Warning] Did not reach target power in tested days")
            print(f"[Tip] Estimated days needed: {estimated_days} days")

    summary = _build_summary(
        start_date_str, end_date_str, mean_treat, sigma,
        consolidated, extra_stats,
        mde=mde, delta_log=delta, delta_pct=delta_pct, delta_abs=delta_abs,
        best_days=best_days, best_power=best_power, estimated_days_needed=estimated_days,
        auto_mde=False
    )
    return {"summary": summary, "power_curve": results_df, "residuals": residuals}


# ── Helper functions ──────────────────────────────────────────────────────

def _print_header(consolidated, cluster_idx, auto_mde):
    """Print the section header based on mode."""
    if consolidated:
        label = "CONSOLIDATED EXPERIMENT MDE" if auto_mde else "CONSOLIDATED EXPERIMENT DURATION"
    elif cluster_idx is not None:
        suffix = " — AUTO-MDE" if auto_mde else ""
        label = f"GEO DURATION ESTIMATION{suffix} (Cluster {cluster_idx})"
    else:
        suffix = " (AUTO-MDE)" if auto_mde else ""
        label = f"GEO DURATION ESTIMATION{suffix}"
    print(f"\n=== {label} ===")


def _print_common_info(treatment_geo, control_geos, start_date_str, end_date_str):
    """Print treatment/control info and evaluating period."""
    print(f"Treatment: {treatment_geo}")
    if control_geos is not None:
        print(f"Control: {control_geos}")

    print("\n=== EVALUATING PERIOD ===")
    print(f"Start Date: {start_date_str}")
    print(f"End Date: {end_date_str}")


def _print_stats(consolidated, sigma, mean_treat, extra_stats):
    """Print statistics section — different for consolidated vs per-cluster."""
    if consolidated:
        method = extra_stats.get("method", "unknown")
        n_series = extra_stats.get("n_series", 1)
        if method == "residuals":
            print("\n=== RESIDUAL VARIANCE ANALYSIS ===")
            print(f"Mean: {mean_treat:.2f}")
            print(f"Std (mean of residuals): {sigma:.4f}")
            print(f"Cluster residual series averaged: {n_series}")
        else:
            print("\n=== TREATMENT VARIANCE ANALYSIS ===")
            print(f"Mean: {mean_treat:.2f}")
            print(f"Std (log-diff): {sigma:.4f}")
            print(f"Treatment geos pooled: {n_series}")
    else:
        print("\n=== REGRESSION STATISTICS ===")
        print(f"Mean: {mean_treat:.2f}")
        print(f"Std Residual: {sigma:.4f}")
        print(f"R-Squared: {extra_stats['r_squared']:.4f}")
        print(f"Correlation: {extra_stats['correlation']:.3f}")


def _build_summary(
    start_date_str, end_date_str, mean_treat, sigma,
    consolidated, extra_stats,
    mde, delta_log, delta_pct, delta_abs,
    best_days, best_power, estimated_days_needed,
    auto_mde
):
    """Build the summary dict for the return value."""
    summary = {
        "start_date": start_date_str,
        "end_date": end_date_str,
        "mean": mean_treat,
        "sigma": sigma,
        "mde": mde,
        "delta_log": delta_log,
        "delta_pct": delta_pct,
        "delta_abs": delta_abs,
        "best_days": best_days,
        "best_power": best_power,
        "estimated_days_needed": estimated_days_needed,
        "auto_mde": auto_mde,
        "consolidated": consolidated,
    }
    summary.update(extra_stats)
    return summary