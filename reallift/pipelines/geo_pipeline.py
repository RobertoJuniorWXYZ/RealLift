import warnings
import pandas as pd
import numpy as np
from ..geo.discovery import discover_geo_clusters
from ..geo.validation import validate_geo_clusters
from ..geo.duration import estimate_duration
from ..geo.synthetic import run_synthetic_control, plot_synthetic_control
from ..geo.placebo import run_placebo_tests, plot_placebo_tests
from ..config.defaults import DEFAULT_POWER, DEFAULT_TREATMENT_PCTS, DEFAULT_EXPERIMENT_DAYS

def run_geo_experiment(
    filepath,
    date_col,
    treatment_start_date,
    treatment_end_date=None,
    doe=None,
    scenario=None,
    start_date=None,
    end_date=None,
    geos=None,
    n_treatment=1,
    fixed_treatment=None,
    mde=0.015,
    experiment_days=[21, 60],
    n_folds=5,
    random_state=None,
    plot=True,
    verbose=True
) -> dict:
    """
    Run a complete GeoLift experiment analysis pipeline.

    Parameters:
        filepath (str): Path to CSV file (including post-intervention data).
        date_col (str): Date column name.
        treatment_start_date (str): Date when the treatment/intervention started.
        treatment_end_date (str): Date when the treatment/intervention ended (analysis window).
        doe (dict): Output from design_of_experiments(). If provided, extracts 
            clusters from the specified scenario.
        scenario (int): Scenario index from DoE to use for analysis.
        start_date (str): YYYY-MM-DD date when analysis begins (optional).
        end_date (str): YYYY-MM-DD date when analysis ends (optional).
        geos (list): List of candidate control geographies (if not using DoE).
        n_treatment (int): Number of treatment groups (if not using DoE).
        fixed_treatment (list): Specific geos to treat (if not using DoE).
        mde (float): Minimum detectable effect expected.
        experiment_days (int or list): Duration range for evaluation.
        n_folds (int): Folds for Cross-Validation (default: 5).
        random_state (int or np.random.Generator): Random seed for reproducibility.
        plot (bool): Whether to plot final graphical diagnostics.
        verbose (bool): Whether to print comprehensive terminal logging results.

    Returns:
        dict: Complete experiment analysis results by cluster and consolidated.
    """
    # 1. Determine clusters
    if doe is not None and scenario is not None:
        if verbose:
            print(f"\n>>> Using experiment design from DoE Scenario {scenario}")
        
        if scenario >= len(doe["scenarios"]):
            raise ValueError(f"Scenario index {scenario} out of range (max {len(doe['scenarios'])-1})")
        
        clusters = doe["scenarios"][scenario]["clusters"]
    else:
        if verbose and not fixed_treatment:
            print("\n>>> No DoE scenario provided. Running cluster discovery...")
        
        clusters = discover_geo_clusters(
            filepath=filepath,
            date_col=date_col,
            geos=geos,
            n_treatment=n_treatment,
            fixed_treatment=fixed_treatment,
            treatment_start_date=treatment_start_date, # Internal discovery only uses pre-split part if this is provided
            start_date=start_date,
            end_date=treatment_start_date, # For results mode, discovery should stop at treatment start
            verbose=verbose
        )

    # 2. Results storage
    results = []
    
    # 2.1 Identify global treatment set to enforce total isolation
    global_treatments = set()
    for c in clusters:
        if isinstance(c["treatment"], list):
            global_treatments.update(c["treatment"])
        else:
            global_treatments.add(c["treatment"])

    # 3. Process each cluster
    for i, cluster in enumerate(clusters):
        # Enforce Isolation: Ensure no other treatment units are in this cluster's control pool
        current_treatment = set(cluster["treatment"]) if isinstance(cluster["treatment"], list) else {cluster["treatment"]}
        exclusive_controls = [g for g in cluster["control"] if g not in global_treatments or g in current_treatment]
        # (Though current treatment shouldn't be in control anyway, this is a safety net)
        
        if verbose:
            print(f"\n" + "-" * 50)
            print(f" ANALYZING CLUSTER {i} ".center(50, "-"))
            print("-" * 50)
            print(f"Treatment: {cluster['treatment']}")
            print(f"Final Exclusive Controls: {exclusive_controls}\n")
            
            # [Warning] if we had to remove contaminated donors
            removed = [g for g in cluster["control"] if g in global_treatments and g not in current_treatment]
            if removed:
                print(f"[Warning] Removed {len(removed)} contaminated donors from this cluster: {removed}")

        # 3.1 Get Validation and Duration metrics (Reuse from DoE or run fresh)
        # ... (lines 94-131 unchanged)
        if doe is not None and scenario is not None:
            # Extract from DoE scenario
            cv_row = doe["scenarios"][scenario]["validation"].iloc[i]
            validation = {"summary": pd.DataFrame([cv_row])}
            
            # Duration (extract specific cluster results)
            dur_full = doe["scenarios"][scenario]["duration"]
            duration = {"summary": dur_full["cluster_results"][i]["summary"]}
            if "mde_curve" in dur_full["cluster_results"][i]:
                duration["mde_curve"] = dur_full["cluster_results"][i]["mde_curve"]
        else:
            # Run fresh validation and duration (legacy fallback)
            validation = validate_geo_clusters(
                filepath=filepath,
                date_col=date_col,
                splits=[cluster],
                treatment_start_date=treatment_start_date,
                start_date=start_date,
                end_date=treatment_start_date, # Validation only looks at pre-period
                n_folds=n_folds,
                cluster_idx=i,
                plot=plot,
                verbose=False # Silent, we print summary later
            )

            duration = estimate_duration(
                filepath=filepath,
                date_col=date_col,
                treatment_geo=cluster["treatment"],
                control_geos=exclusive_controls, # Use exclusive pool
                start_date=start_date,
                end_date=treatment_start_date, 
                mde=mde,
                experiment_days=experiment_days,
                cluster_idx=i,
                verbose=False # Silent
            )

        # 4. Run Analysis (Actual results)
        experiment_type = doe.get("experiment_type", "synthetic_control") if doe is not None else "synthetic_control"
        
        if experiment_type == "matched_did":
            from reallift.geo.did import run_matched_did
            synthetic = run_matched_did(
                filepath=filepath,
                date_col=date_col,
                treatment_geo=cluster["treatment"],
                control_geos=exclusive_controls, # Use exclusive pool
                treatment_start_date=treatment_start_date,
                treatment_end_date=treatment_end_date,
                start_date=start_date,
                end_date=end_date,
                random_state=random_state,
                cluster_idx=i,
                plot=False,
                verbose=verbose
            )
        else:
            synthetic = run_synthetic_control(
                filepath=filepath,
                date_col=date_col,
                treatment_geo=cluster["treatment"],
                control_geos=exclusive_controls, # Use exclusive pool
                treatment_start_date=treatment_start_date,
                treatment_end_date=treatment_end_date,
                start_date=start_date,
                end_date=end_date,
                random_state=random_state,
                cluster_idx=i,
                plot=False,
                verbose=verbose
            )

        # 5. Placebo tests (Significance)
        placebo = run_placebo_tests(
            filepath=filepath,
            date_col=date_col,
            control_geos=cluster["control"],
            treatment_start_date=treatment_start_date,
            observed_pre_mspe=synthetic["pre_mspe"],
            observed_post_mspe=synthetic["post_mspe"],
            observed_lift=synthetic["lift_mean_abs"],
            treatment_end_date=treatment_end_date,
            start_date=start_date,
            end_date=end_date,
            random_state=random_state,
            cluster_idx=i,
            plot=False,
            verbose=verbose,
            experiment_type=experiment_type
        )

        # 6. Final Plots
        if plot:
            if experiment_type == "matched_did":
                from reallift.geo.did import plot_matched_did
                plot_matched_did(
                    df=synthetic["df"],
                    treatment_geo=cluster["treatment"],
                    treatment_idx=synthetic["plotting_data"]["treatment_idx"],
                    y=synthetic["plotting_data"]["y"],
                    synthetic=synthetic["plotting_data"]["synthetic"],
                    effect=synthetic["plotting_data"]["effect"],
                    post_real=synthetic["plotting_data"]["post_real"],
                    post_synth=synthetic["plotting_data"]["post_synth"],
                    effect_pct=synthetic["plotting_data"]["effect_pct"],
                    boot_result=synthetic["bootstrap"]
                )
            else:
                plot_synthetic_control(
                    df=synthetic["df"],
                    treatment_geo=cluster["treatment"],
                    treatment_idx=synthetic["plotting_data"]["treatment_idx"],
                    y=synthetic["plotting_data"]["y"],
                    synthetic=synthetic["plotting_data"]["synthetic"],
                    effect=synthetic["plotting_data"]["effect"],
                    post_real=synthetic["plotting_data"]["post_real"],
                    post_synth=synthetic["plotting_data"]["post_synth"],
                    effect_pct=synthetic["plotting_data"]["effect_pct"],
                    boot_result=synthetic["bootstrap"]
                )
            plot_placebo_tests(
                placebo_ratios=placebo["placebo_ratios"],
                observed_ratio=placebo["observed_ratio"]
            )

        results.append({
            "cluster": cluster,
            "validation": validation,
            "duration": duration,
            "synthetic": synthetic,
            "placebo": placebo
        })

    if verbose and len(results) > 0:
        experiment_type = doe.get("experiment_type", "synthetic_control") if doe is not None else "synthetic_control"
        is_did = experiment_type == "matched_did"
        
        print("\n" + "=" * 70)
        print(" GEO EXPERIMENT RESULTS SUMMARY ".center(70, "="))
        print("=" * 70)
        
        df_first = results[0]["synthetic"]["df"]
        idx_treat = results[0]["synthetic"]["plotting_data"]["treatment_idx"]
        
        pre_start = df_first[date_col].iloc[0].strftime('%Y-%m-%d')
        pre_end = df_first[date_col].iloc[idx_treat-1].strftime('%Y-%m-%d')
        
        # Treatment window (as analyzed)
        post_start = idx_treat
        post_len = len(results[0]["synthetic"]["plotting_data"]["post_real"])
        
        if post_len > 0:
            post_start_date = df_first[date_col].iloc[post_start].strftime('%Y-%m-%d')
            post_end_date = df_first[date_col].iloc[post_start + post_len - 1].strftime('%Y-%m-%d')
            post_days = post_len
            print(f"\nPre-treatment period: {pre_start} → {pre_end}")
            print(f"Intervention period : {post_start_date} → {post_end_date} ({post_days} days)")
        else:
            print(f"\nPre-treatment period: {pre_start} → {pre_end}")
            print(f"Intervention period : N/A (Pre-only analysis)")
        
        n_clusters = len(results)
        num_treated = len(results[0]["cluster"]["treatment"])
        print(f"Clusters analyzed   : {n_clusters} ({num_treated} geo per treatment)")
        
        # Compute MDE per cluster from pre-period residuals
        from scipy.stats import norm as _norm
        _z_alpha = _norm.ppf(1 - 0.05 / 2)  # alpha=0.05
        _z_beta = _norm.ppf(0.80)            # power=80%
        
        TABLE_W = 160
        mde_col_label = f"MDE@{post_days}d" if post_len > 0 else "MDE"
        print("\n" + "-" * TABLE_W)
        print(" CLUSTER-LEVEL INCREMENTAL IMPACT ".center(TABLE_W, "-"))
        print("-" * TABLE_W)
        synth_label = "Matched" if is_did else "Synthetic"
        header = (f"{'Cluster':<8}| {'Treatment':<11}| {'Observed':<11}| {synth_label:<11}"
                  f"| {'Lift (%)':<10}| {'Lift (abs)':<11}| {'CI 95% (%)':<20}| {'CI 95% (abs)':<20}"
                  f"| {'Sig':<8}| {'Causal':<10}| {mde_col_label}")
        print(header)
        print("-" * TABLE_W)
        
        tot_lifts_abs = []
        tot_real_abs = []
        tot_synth_abs = []
        ci_lowers_abs = []
        ci_uppers_abs = []
        
        for i, res in enumerate(results):
            treatment_str = ", ".join(res["cluster"]["treatment"])
            if len(treatment_str) > 10:
                treatment_str = treatment_str[:7] + "..."
                
            syn = res["synthetic"]
            post_synth_sum = syn["plotting_data"]["post_synth"].sum()
            post_real_sum = syn["plotting_data"]["post_real"].sum()
            lift_abs = syn["lift_total"]
            lift_pct = lift_abs / post_synth_sum if post_synth_sum != 0 else 0
            
            ci_l_pct = syn["bootstrap"]["ci_lower_total_pct"]
            ci_u_pct = syn["bootstrap"]["ci_upper_total_pct"]
            ci_l_abs = syn["bootstrap"]["ci_lower_total_abs"]
            ci_u_abs = syn["bootstrap"]["ci_upper_total_abs"]
            
            sig_flag = (ci_l_abs > 0 or ci_u_abs < 0)
            placebo_p = res["placebo"]["p_value"]
            causal_flag = placebo_p <= 0.10
            
            # MDE calculation — matching DoE methodology (log-diff regression with weights)
            df_syn = syn["df"]
            t_idx = syn["plotting_data"]["treatment_idx"]
            treat_geos = res["cluster"]["treatment"]
            ctrl_geos = list(syn["weights"].keys())
            
            try:
                pre_data = df_syn.iloc[:t_idx].copy()
                
                # Build treatment series
                if len(treat_geos) > 1:
                    treat_series = pre_data[treat_geos].mean(axis=1)
                else:
                    treat_series = pre_data[treat_geos[0]]
                
                # Log-diff transformation (same as duration.py)
                cols_for_log = pd.DataFrame({"_treat": treat_series.values}, index=pre_data.index)
                for g in ctrl_geos:
                    cols_for_log[g] = pre_data[g].values
                
                log_diff = np.log(cols_for_log).diff().dropna()
                y_ld = log_diff["_treat"].values
                X_ld = log_diff[ctrl_geos].values
                
                # Apply synthetic control weights (same as duration.py with control_weights)
                w_vals = np.array([syn["weights"][g] for g in ctrl_geos])
                y_pred_ld = X_ld @ w_vals
                sigma = (y_ld - y_pred_ld).std()  # ddof=0, matching DoE
                
                n_days = post_days if post_len > 0 else 21
                delta_log = (_z_alpha + _z_beta) * sigma / np.sqrt(n_days)
                cluster_mde = np.exp(delta_log) - 1
                mde_str = f"{cluster_mde*100:.2f}%"
            except Exception:
                mde_str = "N/A"
            
            lift_pct_str = f"{lift_pct*100:.2f}%"
            ci_str = f"[{ci_l_pct*100:.2f}%, {ci_u_pct*100:.2f}%]"
            ci_abs_str = f"[{ci_l_abs:.1f}, {ci_u_abs:.1f}]"
            
            row = (f"{i:<8}| {treatment_str:<11}| {post_real_sum:<11.2f}| {post_synth_sum:<11.2f}"
                   f"| {lift_pct_str:<10}| {lift_abs:<11.2f}| {ci_str:<20}| {ci_abs_str:<20}"
                   f"| {'[Yes]' if sig_flag else '[No]':<8}| {'[Yes]' if causal_flag else '[No]':<10}| {mde_str}")
            print(row)
            
            tot_lifts_abs.append(lift_abs)
            tot_real_abs.append(post_real_sum)
            tot_synth_abs.append(post_synth_sum)
            ci_lowers_abs.append(ci_l_abs)
            ci_uppers_abs.append(ci_u_abs)
            
        print("-" * TABLE_W)
        
        sum_real = sum(tot_real_abs)
        sum_synth = sum(tot_synth_abs)
        sum_lift = sum_real - sum_synth
        agg_lift_pct = sum_lift / sum_synth if sum_synth != 0 else 0.0
        
        # Proper aggregated bootstrap for confidence intervals
        try:
            from reallift.geo.bootstrap import bootstrap_significance
            
            # Extract time series arrays for the post period
            post_len = len(results[0]["synthetic"]["plotting_data"]["post_real"])
            consolidated_post_real = np.zeros(post_len)
            consolidated_post_synth = np.zeros(post_len)
            
            for res in results:
                consolidated_post_real += res["synthetic"]["plotting_data"]["post_real"]
                consolidated_post_synth += res["synthetic"]["plotting_data"]["post_synth"]
                
            consolidated_effect = consolidated_post_real - consolidated_post_synth
            cons_boot = bootstrap_significance(consolidated_effect, consolidated_post_synth, random_state=random_state)
            
            agg_ci_l_abs = cons_boot["ci_lower_total_abs"]
            agg_ci_u_abs = cons_boot["ci_upper_total_abs"]
            agg_ci_l_pct = cons_boot["ci_lower_total_pct"]
            agg_ci_u_pct = cons_boot["ci_upper_total_pct"]
        except Exception:
            # Fallback to sum of bounds
            agg_ci_l_abs = sum(ci_lowers_abs)
            agg_ci_u_abs = sum(ci_uppers_abs)
            agg_ci_l_pct = agg_ci_l_abs / sum_synth if sum_synth != 0 else 0.0
            agg_ci_u_pct = agg_ci_u_abs / sum_synth if sum_synth != 0 else 0.0
        
        # Consolidated MDE — average of per-cluster residuals (matching DoE consolidated mode)
        try:
            all_cluster_residuals = []
            for res in results:
                syn = res["synthetic"]
                t_idx = syn["plotting_data"]["treatment_idx"]
                treat_geos = res["cluster"]["treatment"]
                ctrl_geos = list(syn["weights"].keys())
                df_syn = syn["df"]
                pre_data = df_syn.iloc[:t_idx].copy()
                
                if len(treat_geos) > 1:
                    treat_series = pre_data[treat_geos].mean(axis=1)
                else:
                    treat_series = pre_data[treat_geos[0]]
                
                cols_for_log = pd.DataFrame({"_treat": treat_series.values}, index=pre_data.index)
                for g in ctrl_geos:
                    cols_for_log[g] = pre_data[g].values
                
                log_diff = np.log(cols_for_log).diff().dropna()
                y_ld = log_diff["_treat"].values
                X_ld = log_diff[ctrl_geos].values
                w_vals = np.array([syn["weights"][g] for g in ctrl_geos])
                residuals = y_ld - X_ld @ w_vals
                all_cluster_residuals.append(pd.Series(residuals))
            
            # Average per-cluster residuals → std of the mean (same as DoE consolidated)
            residuals_df = pd.concat([r.reset_index(drop=True) for r in all_cluster_residuals], axis=1)
            mean_residuals = residuals_df.mean(axis=1)
            sigma_cons = mean_residuals.std()
            
            n_days_cons = post_days if post_len > 0 else 21
            delta_cons = (_z_alpha + _z_beta) * sigma_cons / np.sqrt(n_days_cons)
            consolidated_mde = np.exp(delta_cons) - 1
            cons_mde_str = f"{consolidated_mde*100:.2f}%"
        except Exception:
            cons_mde_str = "N/A"
        
        print(f"\n=== CONSOLIDATED IMPACT ({mde_col_label}: {cons_mde_str}) ===\n")
        expected_label = "Total Matched Baseline (Expected)" if is_did else "Total Synthetic (Expected)"
        print(f"  {'Total Observed Output':<33}: {sum_real:,.2f}")
        print(f"  {expected_label:<33}: {sum_synth:,.2f}")
        print(f"  --------------------------------------------------")
        print(f"  {'INCREMENTAL ABSOLUTE LIFT':<33}: {sum_lift:,.2f}")
        print(f"  {'95% Confidence Interval (abs)':<33}: [{agg_ci_l_abs:,.1f}, {agg_ci_u_abs:,.1f}]")
        print(f"  --------------------------------------------------")
        print(f"  {'INCREMENTAL PERCENTUAL LIFT':<33}: {agg_lift_pct*100:.2f}%")
        print(f"  {'95% Confidence Interval (%)':<33}: [{agg_ci_l_pct*100:.2f}%, {agg_ci_u_pct*100:.2f}%]")
        
        final_sig = "[Yes] Statistically Significant" if (agg_ci_l_abs > 0 or agg_ci_u_abs < 0) else "[No] Not Statistically Significant"
        print(f"\n  Result: {final_sig}\n")
        print("=" * 70 + "\n")

    return {
        "summary": {
            "clusters": clusters
        },
        "results": results
    }




def design_of_experiments(
    filepath,
    date_col,
    start_date=None,
    end_date=None,
    geos=None,
    pct_treatment=None,
    fixed_treatment=None,
    mde=None,
    experiment_days=DEFAULT_EXPERIMENT_DAYS,
    n_folds=5,
    search_mode="ranking",
    experiment_type="synthetic_control",
    use_elasticnet=False,
    n_jobs=None,
    verbose=True
) -> dict:
    """
    Design of Experiments (DoE) — Scenario analysis for GeoLift experiments.

    Automatically generates experiment scenarios at different treatment allocation
    percentages (default: 10%, 20%, 30%), running cluster discovery and duration
    estimation for each. Displays a comparative MDE table to help decide the
    optimal trade-off between sensitivity and intervention cost.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        start_date (str): YYYY-MM-DD date when history begins (optional).
        end_date (str): YYYY-MM-DD date when history ends (optional).
        geos (list): List of candidate geographies. If None, auto-detected from dataset.
        pct_treatment (float or list): Treatment allocation percentage(s).
            - float (e.g. 0.30): single scenario at 30%.
            - list (e.g. [0.10, 0.20, 0.30]): multiple scenarios.
            - None: uses DEFAULT_TREATMENT_PCTS [0.10, 0.20, 0.30].
        fixed_treatment (list): If provided, runs a single scenario with these
            specific geos as treatment (ignores pct_treatment).
        mde (float or None): If None, computes auto-MDE curves. If float, computes
            power curves with fixed MDE.
        experiment_days (int or list): Duration range, e.g. [21, 60].
        n_folds (int): Number of folds for Time Series Cross-Validation (default: 5).
            If 1, uses static temporal split (75/25).
        search_mode (str): Search strategy for treatment selection.
            - "ranking": screen individually, pick top-k (fast, default).
            - "exhaustive": test all C(n,k) combinations (slow, potentially better).
            - "auto": exhaustive if C(n,k) <= 1000, ranking otherwise.
        use_elasticnet (bool): Whether to use ElasticNet to pre-filter controls.
        n_jobs (int): Number of parallel workers for cluster evaluation.
            - 1: sequential (default, safest).
            - -1: use all available CPU cores.
            - N: use N workers (e.g. 4, 8).
        verbose (bool): Whether to print results.

    Returns:
        dict with 'scenarios' list and 'comparison' DataFrame.
    """
    valid_types = ["synthetic_control", "matched_did"]
    if experiment_type not in valid_types:
        raise ValueError(f"Invalid experiment_type '{experiment_type}'. Allowed types are 'synthetic_control' and 'matched_did'.")

    # 1. Detect available geos
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col])

    # Filter by dates
    if start_date is not None:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.to_datetime(end_date)]
    else:
        end_date = df[date_col].max().strftime('%Y-%m-%d')

    if geos is None:
        geos = [col for col in df.columns if col != date_col]
    n_geos = len(geos)

    # 2. Build scenario list
    if fixed_treatment is not None:
        scenario_configs = [{
            "pct": len(fixed_treatment) / n_geos,
            "n_treatment": len(fixed_treatment),
            "fixed": fixed_treatment,
        }]
    else:
        if pct_treatment is None:
            pcts = DEFAULT_TREATMENT_PCTS
        elif isinstance(pct_treatment, (list, tuple)):
            pcts = sorted(pct_treatment)
        else:
            pcts = [pct_treatment]

        scenario_configs = []
        for pct in pcts:
            n_treat = max(1, round(n_geos * pct))
            scenario_configs.append({
                "pct": pct,
                "n_treatment": n_treat,
                "fixed": None,
            })

    pre_start = df[date_col].min().strftime('%Y-%m-%d') if not df.empty else "N/A"

    if verbose:
        print("\n" + "=" * 70)
        print(" DESIGN OF EXPERIMENTS ".center(70, "="))
        print("=" * 70)
        print(f"\nTotal geos available: {n_geos}")
        print(f"Scenarios to evaluate: {len(scenario_configs)}")
        print(f"Search mode: {search_mode}")
        print(f"Pre-treatment period: {pre_start} → {end_date}")
        print(f"Experiment duration: {experiment_days}\n")

    # 3. Global screening (only for ranking mode)
    max_n_treat = max(c["n_treatment"] for c in scenario_configs)
    has_fixed = any(c["fixed"] for c in scenario_configs)

    global_ranking = None
    if not has_fixed and search_mode == "ranking":
        from reallift.geo.discovery import _evaluate_combinations

        if verbose:
            print("Screening all geos individually...")

        df_screen = pd.read_csv(filepath)
        df_screen[date_col] = pd.to_datetime(df_screen[date_col], format='mixed', dayfirst=True, errors='coerce')
        df_screen = df_screen.dropna(subset=[date_col])
        if start_date:
            df_screen = df_screen[df_screen[date_col] >= pd.to_datetime(start_date)]
        if end_date:
            df_screen = df_screen[df_screen[date_col] <= pd.to_datetime(end_date)]
        df_screen = df_screen.groupby(date_col).sum(numeric_only=True).reset_index()
        df_screen = df_screen.sort_values(date_col).reset_index(drop=True)

        alpha_grid = [0.001, 0.01, 0.1]
        l1_grid = [0.2, 0.5, 0.8]

        phase1_combos = [[g] for g in geos]
        phase1_results = _evaluate_combinations(
            df_screen, geos, phase1_combos,
            all_treatment_geos=set(),
            use_elasticnet=use_elasticnet,
            alpha_grid=alpha_grid, l1_grid=l1_grid,
            verbose=verbose, desc="Screening geos", n_jobs=n_jobs
        )
        phase1_results.sort(key=lambda x: x["ser"])
        global_ranking = [r["treatment"][0] for r in phase1_results]

        if verbose:
            print(f"Global ranking (top {max_n_treat}): {global_ranking[:max_n_treat]}\n")

    # 4. Run each scenario
    scenarios = []

    for s_idx, config in enumerate(scenario_configs):
        n_treat = config["n_treatment"]
        pct = config["pct"]

        if verbose:
            print("-" * 70)
            print(f"SCENARIO {s_idx + 1} — {pct:.0%} Treatment ({n_treat} geo{'s' if n_treat > 1 else ''})")
            print("-" * 70)

        # 4a. Find clusters
        try:
            if config["fixed"]:
                # User-specified fixed treatment
                best_groups = [{"treatment": config["fixed"]}]
                search_mode_used = "fixed"
            elif global_ranking is not None:
                # Greedy Sequential Search mode handles discovery inside OOF loop
                best_groups = None
                search_mode_used = "ranking"
            else:
                # Exhaustive mode
                best_groups = discover_geo_clusters(
                    filepath=filepath,
                    date_col=date_col,
                    geos=geos,
                    n_treatment=n_treat,
                    start_date=start_date,
                    end_date=end_date,
                    use_elasticnet=use_elasticnet,
                    search_mode=search_mode,
                    verbose=True,
                    show_results=False,
                    n_jobs=n_jobs
                )
                search_mode_used = "exhaustive"
                if verbose:
                    print(f"  Identified top {len(best_groups)} combinations. Proceeding to OOF Refinement.")
        except Exception as e:
            if verbose:
                print(f"  [Error] Failed to find clusters: {e}\n")
            scenarios.append({
                "pct_treatment": pct,
                "n_treatment": n_treat,
                "clusters": None,
                "duration": None,
                "error": str(e),
            })
            continue

        # 4b. OOF Refinement & Cross-validation
        refined_clusters = []
        cv_rows = []
        df_pre = df.copy()
        
        run_as_matched_did = (experiment_type == "matched_did")

        if global_ranking is not None:
            if run_as_matched_did:
                if verbose: print("\n  [Matched DiD] Evaluating all candidates to maximize correlation...")
                
                all_evals = []
                for candidate in geos:
                    try:
                        candidate_eval = discover_geo_clusters(
                            filepath=filepath, date_col=date_col, geos=geos,
                            fixed_treatment=[candidate], start_date=start_date, end_date=end_date,
                            use_elasticnet=use_elasticnet, verbose=False, show_results=False
                        )
                        current_cluster = candidate_eval[0].copy()
                    except Exception:
                        continue
                    
                    best_cluster, best_cv_row, passed, iters = _run_oof_refinement_single(
                        current_cluster, filepath, date_col, df_pre, start_date, end_date, n_folds, experiment_type=experiment_type
                    )
                    
                    all_evals.append({
                        "raw": current_cluster, "best": best_cluster, "cv": best_cv_row,
                        "candidate": candidate, "r2": best_cv_row["r2_test"], "iters": iters
                    })
                
                # Sort descending by maximum reached holistic R²
                all_evals.sort(key=lambda x: x["r2"], reverse=True)
                
                # Identify the intended treatment set for this scenario based on Matched DiD ranking
                intended_treatment_pool = [item["candidate"] for item in all_evals[:n_treat]]
                
                locked_treatments = []
                
                # Selection Loop: pick candidates and ensure their donor pools are clean 
                # from OTHER treatments in this specific scenario.
                for item in all_evals:
                    if len(refined_clusters) >= n_treat: break
                    candidate = item["candidate"]
                    
                    if candidate in intended_treatment_pool:
                        # PURIFY: Remove any other unit that belongs to the treatment pool from this candidate's donor set
                        best_mod = item["best"].copy()
                        others = [t for t in intended_treatment_pool if t != candidate]
                        
                        clean_controls = [d for d in best_mod["control"] if d not in others]
                        
                        # Update weights to be uniform over the clean pool (DiD Standard)
                        if len(clean_controls) > 0:
                            best_mod["control"] = clean_controls
                            best_mod["control_weights"] = [1.0/len(clean_controls)] * len(clean_controls)
                            
                            item["best"] = best_mod
                            if verbose: print(f"    - Candidate: {candidate:<10} Selected (Cleaned, R² = {item['r2']:.4f})      ")
                            refined_clusters.append(item)
                            locked_treatments.append(candidate)
                        
                # Sort refined_clusters so final output is strict R2 order
                refined_clusters.sort(key=lambda x: x["r2"], reverse=True)
                            
            else:
                if verbose: print("\n  [Greedy Lock Search]\n")
                
                # Initialize: top n_treat candidates from the SER ranking
                current_candidates = list(global_ranking[:n_treat])
                next_rank_idx = n_treat
                
                max_iterations = len(global_ranking)
                locked_clusters = []       # Consolidated clusters (frozen)
                locked_treatments = set()  # Treatment geos already consolidated
                locked_donors = set()      # Donors used by consolidated clusters
                all_failed = []            # All failed candidates across iterations (for fallback)
                
                found = False
                iteration = 0
                # Track iterations that test a single candidate (for compact display)
                single_candidate_buffer = []
                single_candidate_skips = []
                
                def _flush_single_buffer(buf, skips, n_consolidated):
                    """Print compacted single-candidate iteration results."""
                    if not buf:
                        return
                    iter_range = f"Iter {buf[0]['iter']}" if len(buf) == 1 else f"Iter {buf[0]['iter']}–{buf[-1]['iter']}"
                    print(f"    {iter_range} | Testing 1 candidate each | {n_consolidated} consolidated")
                    # Group failures into compact lines (4 per line)
                    failed_items = [f"{r['candidate']} ({r['r2']:.2f})" for r in buf if not r['passed']]
                    passed_items = [r for r in buf if r['passed']]
                    for r in passed_items:
                        print(f"      [Approved] {r['candidate']:<10} R²={r['r2']:.4f}  Gap={r['gap']:.4f}")
                    if failed_items:
                        for j in range(0, len(failed_items), 4):
                            chunk = " | ".join(failed_items[j:j+4])
                            print(f"      [Failed] {chunk}")
                    if skips:
                        skip_names = ", ".join(skips)
                        print(f"      → Skipped {len(skips)} donor-blocked ({skip_names})")
                
                for iteration in range(1, max_iterations + 1):
                    # Only test non-locked candidates
                    candidates_to_test = [c for c in current_candidates if c not in locked_treatments]
                    
                    if not candidates_to_test:
                        break
                    
                    is_single = len(candidates_to_test) == 1
                    
                    iter_results = []
                    
                    for candidate in candidates_to_test:
                        # Exclude all other treatments (locked + other new) from donor pool
                        other_treatments = set(current_candidates) - {candidate}
                        available_geos = [g for g in geos if g not in other_treatments]
                        
                        try:
                            candidate_eval = discover_geo_clusters(
                                filepath=filepath, date_col=date_col, geos=available_geos,
                                fixed_treatment=[candidate], start_date=start_date, end_date=end_date,
                                use_elasticnet=use_elasticnet, verbose=False, show_results=False
                            )
                            current_cluster = candidate_eval[0].copy()
                        except Exception:
                            iter_results.append({"candidate": candidate, "passed": False})
                            continue
                        
                        best_cluster, best_cv_row, passed, iters = _run_oof_refinement_single(
                            current_cluster, filepath, date_col, df_pre, start_date, end_date, n_folds, experiment_type=experiment_type
                        )
                        
                        r2_t = best_cv_row["r2_test"]
                        gap_t = best_cv_row["r2_train"] - r2_t
                        
                        iter_results.append({
                            "candidate": candidate, "passed": passed,
                            "raw": current_cluster, "best": best_cluster, "cv": best_cv_row,
                            "r2": r2_t, "gap": gap_t
                        })
                    
                    # Consolidate (lock) newly passed clusters
                    newly_passed = [r for r in iter_results if r.get("passed", False)]
                    newly_failed = [r for r in iter_results if not r.get("passed", False)]
                    
                    for r in newly_passed:
                        locked_clusters.append(r)
                        locked_treatments.add(r["candidate"])
                        if r.get("best"):
                            locked_donors.update(r["best"].get("control", []))
                    
                    all_failed.extend([r for r in newly_failed if r.get("best") is not None])
                    
                    # Find replacements + track skipped
                    iter_skipped = []
                    new_candidates = sorted(locked_treatments)
                    
                    if len(locked_clusters) < n_treat:
                        while len(new_candidates) < n_treat and next_rank_idx < len(global_ranking):
                            next_geo = global_ranking[next_rank_idx]
                            next_rank_idx += 1
                            if next_geo in locked_treatments:
                                continue
                            if next_geo in locked_donors:
                                iter_skipped.append(next_geo)
                                continue
                            new_candidates.append(next_geo)
                    
                    # --- VERBOSE OUTPUT ---
                    if verbose:
                        if is_single:
                            # Buffer single-candidate iterations for compact display
                            for r in iter_results:
                                single_candidate_buffer.append({
                                    "iter": iteration,
                                    "candidate": r["candidate"],
                                    "passed": r.get("passed", False),
                                    "r2": r.get("r2", 0.0),
                                    "gap": r.get("gap", 0.0),
                                })
                            single_candidate_skips.extend(iter_skipped)
                        else:
                            # Flush any pending single-candidate buffer first
                            _flush_single_buffer(single_candidate_buffer, single_candidate_skips, len(locked_treatments) - sum(1 for r in newly_passed))
                            single_candidate_buffer.clear()
                            single_candidate_skips.clear()
                            
                            # Print multi-candidate iteration
                            cons_str = f" | {len(locked_treatments) - len(newly_passed)} consolidated" if locked_treatments - set(r["candidate"] for r in newly_passed) else ""
                            print(f"    Iter {iteration} | Testing {len(candidates_to_test)} candidates{cons_str}")
                            for r in iter_results:
                                status = "Approved" if r.get("passed") else "Failed  "
                                print(f"      [{status}] {r['candidate']:<10} R²={r.get('r2', 0):.4f}  Gap={r.get('gap', 0):.4f}")
                            
                            n_cons = len(locked_clusters)
                            parts = [f"Consolidated {n_cons}"]
                            if iter_skipped:
                                skip_names = ", ".join(iter_skipped)
                                parts.append(f"Skipped {len(iter_skipped)} donor-blocked ({skip_names})")
                            print(f"      → {' | '.join(parts)}")
                    
                    # Check if all slots are filled
                    if len(locked_clusters) >= n_treat:
                        refined_clusters = [{"raw": r.get("raw"), "best": r["best"], "cv": r["cv"]} for r in locked_clusters[:n_treat]]
                        found = True
                        # Flush remaining buffer
                        if verbose:
                            _flush_single_buffer(single_candidate_buffer, single_candidate_skips, len(locked_treatments) - len(newly_passed))
                            single_candidate_buffer.clear()
                            single_candidate_skips.clear()
                            print(f"\n    [Success] All {n_treat} candidates consolidated by iteration {iteration}.")
                        break
                    
                    if len(new_candidates) < n_treat:
                        # Flush remaining buffer
                        if verbose:
                            _flush_single_buffer(single_candidate_buffer, single_candidate_skips, len(locked_treatments))
                            single_candidate_buffer.clear()
                            single_candidate_skips.clear()
                            print(f"\n    [Exhausted] Ranking exhausted after iteration {iteration}.")
                        break
                    
                    current_candidates = new_candidates
                
                used_fallback = not found
                
                if not found:
                    # Assemble: locked clusters + best failed candidates
                    refined_clusters = [{"raw": r.get("raw"), "best": r["best"], "cv": r["cv"]} for r in locked_clusters]
                    
                    remaining = n_treat - len(refined_clusters)
                    fallback_names = []
                    if remaining > 0 and all_failed:
                        all_failed.sort(key=lambda x: x["cv"]["r2_test"], reverse=True)
                        for r in all_failed[:remaining]:
                            refined_clusters.append({"raw": r.get("raw"), "best": r["best"], "cv": r["cv"]})
                            fallback_names.append(r["candidate"])
                    
                    if verbose:
                        fb_str = f" + {len(fallback_names)} fallback ({', '.join(fallback_names)})" if fallback_names else ""
                        print(f"\n    [Result] {len(locked_clusters)} consolidated{fb_str} to fill {n_treat} slots.")
                
                # ── DESIGN QUALITY ──
                if verbose:
                    r2_vals = []
                    for item in refined_clusters:
                        if item.get("cv") is not None:
                            r2_vals.append(max(item["cv"]["r2_test"], 0.01))
                    
                    if r2_vals:
                        quality_score = len(r2_vals) / sum(1.0 / v for v in r2_vals)
                        if quality_score >= 0.90:
                            rating = "Excellent"
                        elif quality_score >= 0.75:
                            rating = "Good"
                        elif quality_score >= 0.60:
                            rating = "Fair"
                        else:
                            rating = "Poor"
                        
                        n_consolidated = len(locked_clusters)
                        n_fallback = len(refined_clusters) - n_consolidated
                        
                        print(f"\n  DESIGN QUALITY")
                        print(f"  {'─' * 70}")
                        print(f"  Quality Score : {quality_score:.2f} [{rating}]{'':8}(harmonic mean of R² test)")
                        print(f"  Consolidated  : {n_consolidated}/{n_treat} ({n_consolidated/n_treat:.0%}){'':10}(passed strict OOF rules)")
                        if n_fallback > 0:
                            fb_details = []
                            for fn in fallback_names:
                                for item in refined_clusters:
                                    if item.get("best") and item["best"]["treatment"][0] == fn:
                                        fb_details.append(f"{fn}, R² test = {item['cv']['r2_test']:.4f}")
                            fb_str = "; ".join(fb_details) if fb_details else f"{n_fallback} cluster(s)"
                            print(f"  Fallback      : {n_fallback} cluster{'s' if n_fallback > 1 else ''}{'':10}({fb_str})")
                        print(f"  {'─' * 70}")
            
            clusters = [item["best"] for item in refined_clusters]
            cv_rows = [item["cv"] for item in refined_clusters]
            cv_summary = pd.DataFrame(cv_rows).reset_index(drop=True)
            
        else:
            if verbose: print(f"\n  [{'Matched DiD' if experiment_type == 'matched_did' else 'OOF Refinement'}]:")
            
            best_raw_items = None
            best_r2_sum = -float("inf")
            all_passed = False
            
            for group_idx, group in enumerate(best_groups):
                try_geos = group["treatment"]
                if search_mode_used == "exhaustive" and verbose:
                    print(f"\n    [Option {group_idx + 1}/{len(best_groups)}] Evaluating combination: {try_geos}")
                
                # Retrieve individual clusters for this combination
                current_clusters = discover_geo_clusters(
                    filepath=filepath, date_col=date_col, geos=geos,
                    fixed_treatment=try_geos, start_date=start_date, end_date=end_date,
                    use_elasticnet=use_elasticnet, verbose=False, n_jobs=n_jobs
                )
                
                raw_items = []
                group_passed = True
                r2_sum = 0
                
                for i, cluster in enumerate(current_clusters):
                    treat_list = cluster['treatment']
                    treat_str = treat_list[0] if treat_list else "Unknown"
                    eval_cluster = cluster.copy()
                    
                    best_cluster, best_cv_row, passed, iters = _run_oof_refinement_single(
                        eval_cluster, filepath, date_col, df_pre, start_date, end_date, n_folds, experiment_type=experiment_type
                    )
                    
                    r2_t = best_cv_row["r2_test"]
                    gap_t = best_cv_row["r2_train"] - r2_t
                    r2_sum += r2_t
                    
                    if experiment_type == "matched_did":
                        if passed:
                            if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Optimal (R² = {r2_t:.4f}, Iters: {iters})")
                        else:
                            if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Failed strict rules. Best R² = {r2_t:.4f}, Iters: {iters}")
                            group_passed = False
                    else:
                        if passed:
                            if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Optimal (OOF R² = {r2_t:.4f}, Gap = {gap_t:.4f}, Iters: {iters})")
                        else:
                            if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Failed strict rules. (OOF R² = {r2_t:.4f}, Gap = {gap_t:.4f}, Iters: {iters})")
                            group_passed = False
                    
                    raw_items.append({
                        "raw": eval_cluster, "best": best_cluster,
                        "cv": best_cv_row, "passed": passed
                    })
                
                # Keep track of the best group in case all fail
                if best_raw_items is None or r2_sum > best_r2_sum:
                    best_raw_items = raw_items
                    best_r2_sum = r2_sum
                
                if group_passed:
                    all_passed = True
                    if search_mode_used == "exhaustive" and verbose:
                        print(f"    [Success] Found optimal passing combination: {try_geos}")
                    best_raw_items = raw_items
                    break
                else:
                    if search_mode_used == "exhaustive" and verbose:
                        print(f"    [Failed] Combination {try_geos} failed strict rules.")

            if experiment_type == "synthetic_control" and not all_passed:
                if verbose:
                    if search_mode_used == "exhaustive":
                        print("\n  [AUTO SENSOR] [Warning] All top combinations failed strict rules.")
                        print("                  Falling back to the combination with highest overall R².")
                    else:
                        print("\n  [AUTO SENSOR] [Warning] Some fixed clusters failed strict rules. Data may be too volatile for stable Synthetic Control.")
                        print("                  Consider re-evaluating the design with experiment_type='matched_did'.")
                    
            clusters = [item["best"] for item in best_raw_items]
            cv_rows = [item["cv"] for item in best_raw_items]
            cv_summary = pd.DataFrame(cv_rows).reset_index(drop=True)


        # 4c. Estimate duration (per-cluster + consolidated)
        duration = estimate_duration(
            filepath=filepath,
            date_col=date_col,
            clusters=clusters,
            mde=mde,
            experiment_days=experiment_days,
            start_date=start_date,
            end_date=end_date,
            verbose=False
        )

        # 4d. Compact verbose per scenario
        if verbose:
            _print_scenario_table(clusters, duration, mde, cv_summary, total_geos=n_geos, experiment_days=experiment_days, experiment_type=experiment_type)

        treatment_pool = []
        for c in clusters:
            treatment_pool.extend(c["treatment"])

        scenarios.append({
            "pct_treatment": pct,
            "n_treatment": n_treat,
            "treatment_pool": treatment_pool,
            "clusters": clusters,
            "duration": duration,
            "validation": cv_summary,
        })

    # 4. Final comparison table
    comparison = _build_comparison(scenarios, mde, experiment_days)

    if verbose:
        _print_comparison_table(comparison, mde, experiment_days=experiment_days)

    return {
        "experiment_type": experiment_type,
        "scenarios": scenarios,
        "comparison": comparison,
    }


# ── DoE Helpers ───────────────────────────────────────────────────────────

def _run_oof_refinement_single(cluster, filepath, date_col, df_pre, start_date, end_date, n_folds, experiment_type="synthetic_control"):
    import cvxpy as cp
    import numpy as np
    from reallift.geo.validation import validate_geo_clusters

    current_cluster = cluster.copy()
    history = []

    while True:
        if experiment_type == "matched_did":
            # Bypass OOF splitting for Matched DiD, evaluate holistic equal-weighted R² 
            t_cols = current_cluster["treatment"]
            c_cols = current_cluster["control"]
            y_hol = df_pre[t_cols].mean(axis=1).values.astype(float)
            if len(c_cols) > 0:
                X_hol = df_pre[c_cols].mean(axis=1).values.astype(float)
            else:
                X_hol = np.zeros_like(y_hol)
                
            if np.std(y_hol) > 0 and np.std(X_hol) > 0:
                corr_hol = float(np.corrcoef(y_hol, X_hol)[0, 1])
            else:
                corr_hol = 0.0
                
            # Treat r2_holistic identically so OOF rules just validate raw correlation
            r2_hol = corr_hol ** 2
            
            cv_row = pd.Series({
                "treatment": current_cluster["treatment"],
                "r2_train": r2_hol,
                "r2_test": r2_hol,
                "mape_train": 0.0, "mape_test": 0.0,
                "wape_train": 0.0, "wape_test": 0.0
            })
        else:
            validation = validate_geo_clusters(
                filepath=filepath,
                date_col=date_col,
                splits=[current_cluster],
                treatment_start_date=end_date,
                start_date=start_date,
                n_folds=n_folds,
                plot=False,
                verbose=False
            )
            cv_row = validation["summary"].iloc[0]
            
        history.append((current_cluster.copy(), cv_row))

        controls = current_cluster["control"].copy()
        weights = current_cluster.get("control_weights", []).copy()
        
        if len(controls) <= 1:
            break
        
        min_idx = int(np.argmin(weights)) if len(weights) > 0 else -1
        if min_idx >= 0:
            controls.pop(min_idx)
            weights.pop(min_idx)
        else:
            controls.pop()
        
        t_cols = current_cluster["treatment"]
        y_syn = df_pre[t_cols].mean(axis=1).values.astype(float)
        X_syn = df_pre[controls].values.astype(float)
        
        y_mean_syn = y_syn.mean() if y_syn.mean() != 0 else 1e-10
        X_mean_syn = X_syn.mean(axis=0)
        X_mean_syn[X_mean_syn == 0] = 1e-10

        y_norm_syn = y_syn / y_mean_syn
        X_norm_syn = X_syn / X_mean_syn

        w_syn = cp.Variable(len(controls))
        
        # NO INTERCEPT in refinement to prevent leakage absorption
        obj_syn = cp.Minimize(cp.sum_squares(y_norm_syn - (X_norm_syn @ w_syn)))
        cons_syn = [w_syn >= 0, cp.sum(w_syn) == 1]
        prob_syn = cp.Problem(obj_syn, cons_syn)
        
        try:
            prob_syn.solve(solver=cp.SCS, verbose=False)
            w_vals = np.array(w_syn.value).flatten()
            w_vals[w_vals < 0] = 0.0
            sum_w = np.sum(w_vals)
            if sum_w > 0:
                w_vals = w_vals / sum_w
            new_weights = [float(w) for w in w_vals]
            
            # NO INTERCEPT in refinement models
            alpha_val = 0.0 
            y_pred = (X_norm_syn @ np.array(new_weights)) * y_mean_syn
            
            if np.std(y_syn) > 0 and np.std(y_pred) > 0:
                corr = float(np.corrcoef(y_syn, y_pred)[0, 1])
            else:
                corr = 0.0
        except Exception:
            new_weights = [1.0/len(controls)] * len(controls)
            corr = 0.0

        current_cluster["control"] = controls
        current_cluster["control_weights"] = new_weights
        current_cluster["correlation"] = corr

    valid_steps = [
        step for step in history 
        if step[1]["r2_test"] >= 0.6 
        and step[1]["r2_train"] >= 0.6 
        and abs(step[1]["r2_train"] - step[1]["r2_test"]) <= 0.20
    ]

    if valid_steps:
        valid_steps.sort(key=lambda x: x[1]["r2_test"], reverse=True)
        best_cluster, best_cv_row = valid_steps[0]
        passed = True
    else:
        history.sort(key=lambda x: x[1]["r2_test"], reverse=True)
        best_cluster, best_cv_row = history[0]
        passed = False
        
    if experiment_type == "matched_did":
        controls = best_cluster["control"]
        if len(controls) > 0:
            best_cluster["control_weights"] = [1.0 / len(controls)] * len(controls)
        
    return best_cluster, best_cv_row, passed, len(history)

def _print_scenario_table(clusters, duration, mde, cv_summary=None, total_geos=None, experiment_days=None, experiment_type=None):
    """Print compact per-cluster table for a scenario with CV metrics and donor pools."""
    cluster_results = duration["cluster_results"]
    consolidated = duration["consolidated"]

    is_auto = mde is None

    # ── Duration & MDE Grid ──
    if is_auto:
        days_to_print = experiment_days if experiment_days else [21, 30, 60]
        mde_cols = [f"MDE @{d}d" for d in days_to_print]
        mde_hdr = " | ".join(f"{col:<9}" for col in mde_cols)
        print(f"\n  {'Cluster':<7} | {'Treatment':<10} | {'Controls':<8} | {mde_hdr}")
        print("  " + "-" * (35 + len(mde_hdr)))

        for i, (cl, cr) in enumerate(zip(clusters, cluster_results)):
            treat = ", ".join(cl["treatment"])
            if len(treat) > 10:
                treat = treat[:7] + "..."
            weights = cl.get("control_weights", [])
            n_ctrl = sum(1 for w in weights if w > 0.001) if weights else len(cl["control"])
            
            curve = cr.get("mde_curve")

            mde_strs = []
            for d in days_to_print:
                if curve is not None:
                    val = curve.loc[curve["days"] == d, "mde"]
                    mde_strs.append(f"{val.values[0]*100:.2f}%" if len(val) > 0 else "N/A")
                else:
                    mde_strs.append("N/A")

            mde_row = " | ".join(f"{s:<9}" for s in mde_strs)
            print(f"  {i:<7} | {treat:<10} | {n_ctrl:<8} | {mde_row}")

        # Consolidated row
        c_summary = consolidated["summary"]
        c_curve = consolidated.get("mde_curve")
        c_mdes = []
        for d in days_to_print:
            if c_curve is not None:
                val = c_curve.loc[c_curve["days"] == d, "mde"]
                c_mdes.append(f"{val.values[0]*100:.2f}%" if len(val) > 0 else "N/A")
            else:
                c_mdes.append("N/A")

        c_mdes_str = " | ".join(f"{s:<9}" for s in c_mdes)
        print("  " + "-" * (35 + len(mde_hdr)))
        print(f"  {'CONSOL.':<7} | {'pooled':<10} | {'':<8} | {c_mdes_str}")
    else:
        print(f"\n  {'Cluster':<7} | {'Treatment':<10} | {'Controls':<8} | {'Min Days':<8} | {'Power':<7}")
        print("  " + "-" * 51)

        for i, (cl, cr) in enumerate(zip(clusters, cluster_results)):
            treat = ", ".join(cl["treatment"])
            if len(treat) > 10:
                treat = treat[:7] + "..."
            weights = cl.get("control_weights", [])
            n_ctrl = sum(1 for w in weights if w > 0.001) if weights else len(cl["control"])

            best_days = cr["summary"].get("best_days")
            best_power = cr["summary"].get("best_power")

            days_str = f"{int(best_days)}d" if best_days else "N/A"
            power_str = f"{best_power:.1%}" if best_power else "N/A"

            print(f"  {i:<7} | {treat:<10} | {n_ctrl:<8} | {days_str:<8} | {power_str:<7}")

        c_summary = consolidated["summary"]
        c_best = c_summary.get("best_days")
        c_power = c_summary.get("best_power")
        c_days_str = f"{int(c_best)}d" if c_best else "N/A"
        c_power_str = f"{c_power:.1%}" if c_power else "N/A"

        print("  " + "-" * 51)
        print(f"  {'CONSOL.':<7} | {'pooled':<10} | {'':<8} | {c_days_str:<8} | {c_power_str:<7}")

    # ── Treatment & Donor Layout ──
    all_treatments = set()
    all_controls = set()
    for cl in clusters:
        all_treatments.update(cl["treatment"])
        all_controls.update(cl["control"])
    
    distinct_geos = len(all_treatments | all_controls)
    
    print("\n  EXPERIMENTAL SCOPE")
    coverage_str = f"{distinct_geos / total_geos:.0%}" if total_geos else "N/A"
    print(f"  Distinct Geos Used   : {distinct_geos} ({coverage_str} coverage)")
    
    print(f"\n  TEST POOL (TREATMENT UNITS): {', '.join(sorted(list(all_treatments)))}")
    
    print(f"\n  CONTROL DESIGN (DONOR POOL & WEIGHTS)")
    # Build transposed grid: donors (rows) × clusters (columns)
    all_donors_set = set()
    cluster_donor_maps = []
    cluster_treat_names = []
    for cl in clusters:
        treat_name = ", ".join(cl["treatment"])
        if len(treat_name) > 8:
            treat_name = treat_name[:5] + ".."
        cluster_treat_names.append(treat_name)
        donor_map = {}
        controls = cl["control"]
        weights = cl.get("control_weights", [])
        for d, w in zip(controls, weights):
            if w > 0.001:
                donor_map[d] = w
                all_donors_set.add(d)
        cluster_donor_maps.append(donor_map)
    
    all_donors_sorted = sorted(all_donors_set)
    col_w = 8  # column width per cluster
    
    # Header
    hdr = f"  {'Donor':<10} |"
    for tn in cluster_treat_names:
        hdr += f" {tn:>{col_w}} |"
    print(hdr)
    sep = f"  {'─' * 10}─┼" + "─".join(f"{'─' * (col_w + 1)}┼" for _ in cluster_treat_names)
    sep = sep.rstrip("┼") + "┤"
    print(sep)
    
    # Data rows
    for donor in all_donors_sorted:
        row = f"  {donor:<10} |"
        for dm in cluster_donor_maps:
            if donor in dm:
                row += f" {dm[donor]:>{col_w}.3f} |"
            else:
                row += f" {'—':>{col_w}} |"
        print(row)
    
    # Footer: donor count
    print(sep)
    footer = f"  {'Donors':<10} |"
    for dm in cluster_donor_maps:
        footer += f" {len(dm):>{col_w}} |"
    print(footer)

    # ── Cross-Validation Grid ──
    warnings_list = []  # Collect warnings for dedicated section
    if cv_summary is not None and not cv_summary.empty:
        is_pure_did = experiment_type == "matched_did"
        
        if is_pure_did:
            print(f"\n  {'Cluster':<7} | {'Treatment':<10} | {'R²':<17}")
            print("  " + "-" * 38)

            for i, row in cv_summary.iterrows():
                treat = ", ".join(row["treatment"]) if isinstance(row["treatment"], list) else str(row["treatment"])
                if len(treat) > 10:
                    treat = treat[:7] + "..."

                r2 = f"{row['r2_test']:.4f}"

                print(f"  {i:<7} | {treat:<10} | {r2:<17}")
        else:
            print(f"\n  CROSS-VALIDATION SUMMARY")
            print(f"  {'Cluster':<7} | {'Treatment':<10} | {'R² Train':<8} | {'R² Test':<8} | {'MAPE Tr':<8} | {'MAPE Te':<8} | {'WAPE Tr':<8} | {'WAPE Te':<8}")
            print("  " + "-" * 80)

            for i, row in cv_summary.iterrows():
                treat = ", ".join(row["treatment"]) if isinstance(row["treatment"], list) else str(row["treatment"])
                if len(treat) > 10:
                    treat = treat[:7] + "..."

                r2_tr = f"{row['r2_train']:.4f}"
                r2_te = f"{row['r2_test']:.4f}"
                mape_tr = f"{row['mape_train']:.4f}"
                mape_te = f"{row['mape_test']:.4f}"
                wape_tr = f"{row['wape_train']:.4f}"
                wape_te = f"{row['wape_test']:.4f}"

                # Track warnings (don't print inline)
                gap = row['r2_train'] - row['r2_test']
                if row['r2_test'] < 0.60:
                    warnings_list.append(f"  {treat:<10} R² test = {row['r2_test']:.4f} (below 0.60 threshold)")
                elif row['r2_train'] < 0.60:
                    warnings_list.append(f"  {treat:<10} R² train = {row['r2_train']:.4f} (below 0.60 threshold)")
                elif abs(gap) > 0.2:
                    warnings_list.append(f"  {treat:<10} R² gap = {gap:.4f} (instability risk)")

                print(f"  {i:<7} | {treat:<10} | {r2_tr:<8} | {r2_te:<8} | {mape_tr:<8} | {mape_te:<8} | {wape_tr:<8} | {wape_te:<8}")
    
    # ── Dedicated WARNINGS Section ──
    if warnings_list or (experiment_type == "synthetic_control" and any(
        r.get("r2_test", 1.0) < 0.60 if isinstance(r, dict) else (r["r2_test"] < 0.60 if "r2_test" in r.index else False)
        for r in (cv_summary.iloc[i] for i in range(len(cv_summary))) if cv_summary is not None
    )):
        print(f"\n  WARNINGS")
        print(f"  {'─' * 70}")
        if warnings_list:
            n_warn = len(warnings_list)
            print(f"  [Warning] {n_warn} cluster{'s' if n_warn > 1 else ''} with quality concerns:")
            for w in warnings_list:
                print(f"            -{w}")
        
        # Check if any cluster has low R² and suggest alternatives
        has_low_quality = any(
            row["r2_test"] < 0.60
            for _, row in cv_summary.iterrows()
        ) if cv_summary is not None and not cv_summary.empty else False
        
        if has_low_quality:
            n_clusters = len(clusters) if clusters else 0
            print(f"  [Warning] Data may be too volatile for {n_clusters} simultaneous Synthetic")
            print(f"            Control clusters. Consider alternatives:")
            print(f"            - Try search_mode='exhaustive' for optimal partitioning")
            print(f"            - Reduce number of treatment geos")
            print(f"            - Use experiment_type='matched_did'")
        print(f"  {'─' * 70}")

    print()


def _build_comparison(scenarios, mde, experiment_days):
    """Build summary table comparing multiple scenarios."""
    rows = []
    for s_idx, s in enumerate(scenarios):
        if s.get("clusters") is None:
            continue
        
        pct = s.get("pct_treatment", 0)
        n_treat = s["n_treatment"]
        consolidated = s["duration"]["consolidated"]["summary"]
        
        # Calculate distinct geos
        all_treat = set()
        all_ctrl = set()
        for cl in s["clusters"]:
            all_treat.update(cl["treatment"])
            all_ctrl.update(cl["control"])
        distinct_geos = len(all_treat | all_ctrl)

        row = {
            "Scenario": s_idx + 1,
            "% Treated": f"{pct:.0%}",
            "Clusters": len(s["clusters"]),
            "Distinct": distinct_geos,
            "sigma": consolidated["sigma"]
        }

        if mde is None:
            # Auto-MDE mode
            curve = s["duration"]["consolidated"].get("mde_curve")
            days_to_eval = experiment_days if experiment_days else [21, 30, 60]
            for d in days_to_eval:
                if curve is not None:
                    val = curve.loc[curve["days"] == d, "mde"]
                    row[f"mde_{d}d"] = f"{val.values[0]*100:.2f}%" if len(val) > 0 else "N/A"
                else:
                    row[f"mde_{d}d"] = "N/A"
        else:
            # Fixed MDE mode
            best_days = consolidated.get("best_days")
            best_power = consolidated.get("best_power")
            row["best_days"] = f"{int(best_days)}d" if best_days else "N/A"
            row["best_power"] = f"{best_power:.1%}" if best_power else "N/A"

        rows.append(row)

    return pd.DataFrame(rows)


def _print_comparison_table(comparison_df, mde, experiment_days=None):
    """Print the final consolidated comparison cross-scenario."""
    if comparison_df.empty:
        print("\n  No valid scenarios found to compare.")
        return

    print("\n" + "=" * 85)
    print(" EXPERIMENT DESIGN COMPARISON ".center(85, "="))
    print("=" * 85)
    print("")
    
    is_auto = mde is None
    if is_auto:
        days_to_print = experiment_days if experiment_days else [21, 30, 60]
        mde_cols = [f"MDE @{d}d" for d in days_to_print]
        mde_hdr = " | ".join(f"{col:<9}" for col in mde_cols)
        
        header = f"  {'Scenario':<8} | {'% Treated':<10} | {'Clusters':<8} | {'Distinct':<8} | {mde_hdr}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in comparison_df.iterrows():
            mde_vals = [f"{row.get(f'mde_{d}d', 'N/A'):<9}" for d in days_to_print]
            mde_row = " | ".join(mde_vals)
            print(f"  {int(row['Scenario']):<8} | {row['% Treated']:<10} | {int(row['Clusters']):<8} | {int(row['Distinct']):<8} | {mde_row}")
    else:
        header = f"  {'Scenario':<8} | {'% Treated':<10} | {'Clusters':<8} | {'Distinct':<8} | {'Min Days':<10} | {'Power':<7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in comparison_df.iterrows():
            print(f"  {int(row['Scenario']):<8} | {row['% Treated']:<10} | {int(row['Clusters']):<8} | {int(row['Distinct']):<8} | {row['best_days']:<10} | {row['best_power']:<7}")
    
    print("\n" + "=" * 85)

    print("\n" + "=" * 70)
