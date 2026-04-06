import warnings
import pandas as pd
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

    # 3. Process each cluster
    for i, cluster in enumerate(clusters):
        if verbose:
            print(f"\n" + "-" * 50)
            print(f" ANALYZING CLUSTER {i} ".center(50, "-"))
            print("-" * 50)
            print(f"Treatment: {cluster['treatment']}")
            print(f"Control: {cluster['control']}\n")

        # 3.1 Get Validation and Duration metrics (Reuse from DoE or run fresh)
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
                control_geos=cluster["control"],
                start_date=start_date,
                end_date=treatment_start_date, # Duration only looks at pre-period
                mde=mde,
                experiment_days=experiment_days,
                cluster_idx=i,
                verbose=False # Silent
            )

        # 4. Run synthetic control (Actual results)
        synthetic = run_synthetic_control(
            filepath=filepath,
            date_col=date_col,
            treatment_geo=cluster["treatment"],
            control_geos=cluster["control"],
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
            verbose=verbose
        )

        # 6. Final Plots
        if plot:
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
        print("-" * 70)
        
        print("\n=== MODEL ROBUSTNESS (PRE-TREATMENT) ===\n")

        is_auto_mde = results[0]["duration"]["summary"].get("auto_mde", False)
        if is_auto_mde:
            print(f"{'Cluster':<7} | {'Corr':<6} | {'OOF R2':<7} | {'MAPE':<6} | {'MDE @21d':<9} | {'MDE @30d':<9} | {'MDE @60d':<9}")
            print("-" * 85)
        else:
            print(f"{'Cluster':<7} | {'Corr':<6} | {'OOF R2':<7} | {'MAPE':<6} | {'MDE':<6} | {'Min Days':<10} | {'Power'}")
            print("-" * 75)
        
        all_robust = True
        for i, res in enumerate(results):
            corr = res["duration"]["summary"].get("correlation", 0)
            oof_r2 = res["validation"]["summary"].iloc[0]["r2_test"]
            mape_val = res["validation"]["summary"].iloc[0]["mape_test"]

            if oof_r2 <= 0.6: all_robust = False

            if is_auto_mde:
                mde_curve = res["duration"]["mde_curve"]
                mdes = []
                for d in [21, 30, 60]:
                    val = mde_curve.loc[mde_curve["days"] == d, "mde"]
                    mdes.append(f"{val.values[0]*100:.2f}%" if len(val) > 0 else "N/A")
                print(f"{i:<7} | {corr:<6.3f} | {oof_r2:<7.4f} | {mape_val:<6.4f} | {mdes[0]:<9} | {mdes[1]:<9} | {mdes[2]:<9}")
            else:
                mde_val = res["duration"]["summary"]["mde"] * 100
                best_days = res["duration"]["summary"]["best_days"]
                best_power = res["duration"]["summary"]["best_power"]
                days_str = f"{int(best_days)}d" if best_days else "N/A"
                power_str = f"{best_power:.1%}" if best_power else "N/A"
                print(f"{i:<7} | {corr:<6.3f} | {oof_r2:<7.4f} | {mape_val:<6.4f} | {mde_val:<5.2f}% | {days_str:<10} | {power_str}")
            
        if all_robust:
            print("\n✔ Strong predictive performance across all clusters. Results are statistically reliable.")
        else:
            print("\n⚠️ Note: Some clusters show lower predictive robustness (OOF R2 <= 0.6). Interpret with caution.")
            
        print("\n" + "-" * 125)
        print(" CLUSTER-LEVEL INCREMENTAL IMPACT ".center(125, "-"))
        print("-" * 125)
        print(f"{'Cluster':<7} | {'Treatment':<10} | {'Observed':<10} | {'Synthetic':<10} | {'Lift (%)':<8} | {'Lift (abs)':<10} | {'CI 95% (%)':<18} | {'CI 95% (abs)':<18} | {'Sig'}")
        print("-" * 125)
        
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
            
            sig = "✔" if (ci_l_abs > 0 or ci_u_abs < 0) else " "
            ci_str = f"[{ci_l_pct*100:.2f}%, {ci_u_pct*100:.2f}%]"
            ci_abs_str = f"[{ci_l_abs:.1f}, {ci_u_abs:.1f}]"
            
            print(f"{i:<7} | {treatment_str:<10} | {post_real_sum:<10.2f} | {post_synth_sum:<10.2f} | {lift_pct*100:<7.2f}% | {lift_abs:<10.2f} | {ci_str:<18} | {ci_abs_str:<18} | {sig}")
            
            tot_lifts_abs.append(lift_abs)
            tot_real_abs.append(post_real_sum)
            tot_synth_abs.append(post_synth_sum)
            ci_lowers_abs.append(ci_l_abs)
            ci_uppers_abs.append(ci_u_abs)
            
        print("-" * 125)
        
        sum_real = sum(tot_real_abs)
        sum_synth = sum(tot_synth_abs)
        sum_lift = sum_real - sum_synth
        agg_lift_pct = sum_lift / sum_synth if sum_synth != 0 else 0.0
        
        agg_ci_l_abs = sum(ci_lowers_abs)
        agg_ci_u_abs = sum(ci_uppers_abs)
        agg_ci_l_pct = agg_ci_l_abs / sum_synth if sum_synth != 0 else 0.0
        agg_ci_u_pct = agg_ci_u_abs / sum_synth if sum_synth != 0 else 0.0
        
        print("\n=== CONSOLIDATED IMPACT ===\n")
        print(f"  Total Observed Output          : {sum_real:,.2f}")
        print(f"  Total Synthetic (Expected)     : {sum_synth:,.2f}")
        print(f"  --------------------------------------------------")
        print(f"  INCREMENTAL ABOLUTE LIFT       : {sum_lift:,.2f}")
        print(f"  95% Confidence Interval (abs)  : [{agg_ci_l_abs:,.1f}, {agg_ci_u_abs:,.1f}]")
        print(f"  --------------------------------------------------")
        print(f"  INCREMENTAL PERCENTUAL LIFT    : {agg_lift_pct*100:.2f}%")
        print(f"  95% Confidence Interval (%)    : [{agg_ci_l_pct*100:.2f}%, {agg_ci_u_pct*100:.2f}%]")
        
        final_sig = "✔ Statistically Significant" if (agg_ci_l_abs > 0 or agg_ci_u_abs < 0) else "✘ Not Statistically Significant"
        print(f"\n  Result: {final_sig}\n")
        print("=" * 70 + "\n")

    return {
        "summary": {
            "clusters": clusters
        },
        "results": results
    }

def run_geo_requirements(
    filepath,
    date_col,
    start_date=None,
    end_date=None,
    n_treatment=1,
    fixed_treatment=None,
    n_folds=5,
    mde=0.015,
    experiment_days=[21, 60],
    verbose=True
) -> dict:
    """
    Consolidate GEO experiment requirements: finds clusters, validates them, and estimates duration.
    Produces a structured report for each cluster found.

    .. deprecated::
        Use `design_of_experiments()` instead, which provides multi-scenario comparison,
        global screening, and a more comprehensive experiment design workflow.
    """
    warnings.warn(
        "run_geo_requirements() is deprecated. Use design_of_experiments() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # 1. Find best clusters
    clusters = discover_geo_clusters(
        filepath=filepath,
        date_col=date_col,
        n_treatment=n_treatment,
        fixed_treatment=fixed_treatment,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose
    )

    # 2. Validate clusters (silent, we'll format the output manually)
    validation_results = validate_geo_clusters(
        filepath=filepath,
        date_col=date_col,
        splits=clusters,
        treatment_start_date=end_date,
        start_date=start_date,
        n_folds=n_folds,
        plot=False,
        verbose=False # Silent
    )
    cv_summary = validation_results["summary"]

    final_results = []

    for i, cluster in enumerate(clusters):
        # 3. Estimate duration (silent)
        duration_results = estimate_duration(
            filepath=filepath,
            date_col=date_col,
            treatment_geo=cluster["treatment"], # Pass whole list
            control_geos=cluster["control"],
            start_date=start_date,
            end_date=end_date,
            mde=mde,
            experiment_days=experiment_days,
            verbose=False # Silent
        )
        stats = duration_results["summary"]

        # 4. Extract CV metrics for this specific cluster
        cv_metric = cv_summary.iloc[i]

        if verbose:
            print(f"\n=== GEO EXPERIMENT REQUIREMENTS (Cluster {i}) ===")
            print(f"Treatment: {cluster['treatment']}")
            print(f"Control: {cluster['control']}")

            print(f"\n=== EVALUATING PERIOD ===")
            print(f"Start Date: {cv_metric['start_date']}")
            print(f"End Date: {cv_metric['end_date']}")            

            print(f"\n=== CROSS-VALIDATION ({n_folds} FOLDS) ===")
            print(f"Train R2: {cv_metric['r2_train']:.4f} | OOF R2: {cv_metric['r2_test']:.4f}")
            print(f"Train MAPE: {cv_metric['mape_train']:.4f} | OOF MAPE: {cv_metric['mape_test']:.4f}")
            print(f"Train WAPE: {cv_metric['wape_train']:.4f} | OOF WAPE: {cv_metric['wape_test']:.4f}")

            print(f"\n=== REGRESSION STATISTICS ===")
            print(f"Mean: {stats['mean']:.2f}")
            print(f"Std Residual: {stats['sigma']:.4f}")
            print(f"R-Squared: {stats['r_squared']:.4f}")
            print(f"Correlation: {stats['correlation']:.3f}")

            if stats.get("auto_mde", False):
                mde_curve = duration_results["mde_curve"]
                print(f"\n=== MDE CURVE (power={DEFAULT_POWER:.0%}) ===")
                print(f"{'Days':<6} | {'MDE':<16}")
                print("-" * 26)
                display_days = [d for d in mde_curve["days"] if d % 7 == 0 or d == mde_curve["days"].iloc[0] or d == mde_curve["days"].iloc[-1]]
                for _, row in mde_curve[mde_curve["days"].isin(display_days)].iterrows():
                    print(f"{int(row['days']):<6} | {row['mde']*100:<15.2f}%")
            else:
                print(f"\n=== MDE ===")
                print(f"MDE: {mde*100:.2f}%")
                print(f"Effect absolute: {stats['delta_abs']:.2f}")
                print(f"Effect percent real: {stats['delta_pct']*100:.2f}%")

                print(f"\n=== RESULT ===")
                if stats['best_days']:
                    print(f"✔ Min duration: {int(stats['best_days'])} days")
                    print(f"✔ Power: {stats['best_power']:.2%}")
                else:
                    print("⚠️ Did not reach target power in tested days")
                    print(f"👉 Estimated duration needed: {stats['estimated_days_needed']} days")
            
            print("") # Extra newline for spacing
        
        result_entry = {
            "cluster": cluster,
            "validation": cv_metric.to_dict(),
            "duration": stats,
            "duration_raw": duration_results,
        }
        if stats.get("auto_mde", False):
            result_entry["mde_curve"] = duration_results["mde_curve"]
        final_results.append(result_entry)

    # Collect all treatments and controls (needed for consolidated calc regardless of verbose)
    all_treatments = set()
    all_controls = set()
    for res in final_results:
        all_treatments.update(res["cluster"]["treatment"])
        all_controls.update(res["cluster"]["control"])
    all_treatments_sorted = sorted(list(all_treatments))
    all_controls_sorted = sorted(list(all_controls))

    is_auto_mde = final_results[0]["duration"].get("auto_mde", False) if final_results else False

    if verbose:
        print("\n=== GEO EXPERIMENT REQUIREMENT SUMMARY ===\n")

        if is_auto_mde:
            print(f"{'Cluster':<7} | {'Treatment':<12} | {'Correlation':<11} | {'OOF R2':<6} | {'MAPE':<6} | {'MDE @21d':<9} | {'MDE @30d':<9} | {'MDE @60d':<9}")
            print("-" * 85)
        else:
            print(f"{'Cluster':<7} | {'Treatment':<12} | {'Correlation':<11} | {'OOF R2':<6} | {'MAPE':<6} | {'MDE':<5} | {'Min Duration':<12} | {'Power':<5}")
            print("-" * 80)
        
        max_duration = 0
        all_durations_found = True
        
        for i, res in enumerate(final_results):
            treatment_str = ", ".join(res["cluster"]["treatment"])
            if len(treatment_str) > 12:
                treatment_str = treatment_str[:9] + "..."
            
            corr = res["duration"].get("correlation", 0)
            oof_r2 = res["validation"]["r2_test"]
            mape_val = res["validation"]["mape_test"]

            if is_auto_mde:
                mde_curve = res.get("mde_curve")
                if mde_curve is not None:
                    mde_21 = mde_curve.loc[mde_curve["days"] == 21, "mde"]
                    mde_30 = mde_curve.loc[mde_curve["days"] == 30, "mde"]
                    mde_60 = mde_curve.loc[mde_curve["days"] == 60, "mde"]
                    mde_21_str = f"{mde_21.values[0]*100:.2f}%" if len(mde_21) > 0 else "N/A"
                    mde_30_str = f"{mde_30.values[0]*100:.2f}%" if len(mde_30) > 0 else "N/A"
                    mde_60_str = f"{mde_60.values[0]*100:.2f}%" if len(mde_60) > 0 else "N/A"
                else:
                    mde_21_str = mde_30_str = mde_60_str = "N/A"
                print(f"{i:<7} | {treatment_str:<12} | {corr:<11.4f} | {oof_r2:<6.4f} | {mape_val:<6.4f} | {mde_21_str:<9} | {mde_30_str:<9} | {mde_60_str:<9}")
            else:
                mde_val = res["duration"]["mde"] * 100
                min_dur = res["duration"]["best_days"]
                if min_dur:
                    dur_str = f"{int(min_dur)}d"
                    power_str = f"{res['duration']['best_power']*100:.2f}%"
                    max_duration = max(max_duration, int(min_dur))
                else:
                    dur_str = "N/A"
                    power_str = "N/A"
                    all_durations_found = False
                print(f"{i:<7} | {treatment_str:<12} | {corr:<11.4f} | {oof_r2:<6.4f} | {mape_val:<6.4f} | {mde_val:<4.2f}% | {dur_str:<12} | {power_str:<5}")
            
        print("\n=== CLUSTER ASSIGNMENTS ===\n")
        print(f"{'Cluster':<7} | {'Treatment':<12} | {'Controls'}")
        print("-" * 80)
        for i, res in enumerate(final_results):
            treatment_str = ", ".join(res["cluster"]["treatment"])
            if len(treatment_str) > 12:
                treatment_str = treatment_str[:9] + "..."
            control_str = ", ".join(res["cluster"]["control"])
            print(f"{i:<7} | {treatment_str:<12} | {control_str}")
            
        print("\n\n=== RECOMMENDED EXPERIMENT SETUP ===\n")
        
        print(f"Selected Geos For Treatment: {', '.join(all_treatments_sorted)}")
        print(f"Selected Geos For Control: {', '.join(all_controls_sorted)}\n")
        
        if not is_auto_mde and all_durations_found and max_duration > 0:
            print(f"Recommended duration for intervention: {max_duration} days")
            print("\nRationale:")
            print("Duration chosen to satisfy statistical power requirements across all clusters.")
            print("Ensures consistent measurement and comparability.\n")
        elif not is_auto_mde:
            print("Recommended duration for intervention: N/A")
            print("\nRationale:")
            print("One or more clusters did not reach the minimum statistical power within the maximum tested days.")
            print("Consider increasing the MDE, the experiment_days limit, or relaxing the fixed geographical splits.\n")

    # 5. Consolidated MDE (residual variance analysis)
    cluster_residuals = [
        res["duration_raw"]["residuals"]
        for res in final_results
        if res.get("duration_raw") and res["duration_raw"].get("residuals") is not None
    ]

    consolidated_duration = estimate_duration(
        filepath=filepath,
        date_col=date_col,
        treatment_geo=all_treatments_sorted,
        start_date=start_date,
        end_date=end_date,
        mde=mde,
        experiment_days=experiment_days,
        consolidated=True,
        cluster_residuals=cluster_residuals if cluster_residuals else None,
        verbose=verbose
    )

    return {
        "clusters": clusters,
        "results": final_results,
        "consolidated_duration": consolidated_duration
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
        verbose (bool): Whether to print results.

    Returns:
        dict with 'scenarios' list and 'comparison' DataFrame.
    """
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
            use_elasticnet=True,
            alpha_grid=alpha_grid, l1_grid=l1_grid,
            verbose=verbose, desc="Screening geos"
        )
        phase1_results.sort(key=lambda x: x["synthetic_error_ratio"])
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
                clusters = discover_geo_clusters(
                    filepath=filepath,
                    date_col=date_col,
                    geos=geos,
                    fixed_treatment=config["fixed"],
                    start_date=start_date,
                    end_date=end_date,
                    verbose=False
                )
            elif global_ranking is not None:
                # Ranking mode: use pre-screened top-k
                clusters = discover_geo_clusters(
                    filepath=filepath,
                    date_col=date_col,
                    geos=geos,
                    fixed_treatment=global_ranking[:n_treat],
                    start_date=start_date,
                    end_date=end_date,
                    verbose=False
                )
            else:
                # Exhaustive/auto mode: find best GROUP, then build individual clusters
                best_groups = discover_geo_clusters(
                    filepath=filepath,
                    date_col=date_col,
                    geos=geos,
                    n_treatment=n_treat,
                    start_date=start_date,
                    end_date=end_date,
                    search_mode=search_mode,
                    verbose=True,
                    show_results=False
                )
                # Extract treatment geos from the top-1 recommendation
                best_treatment_geos = best_groups[0]["treatment"]
                if verbose:
                    print(f"  Best combination: {best_treatment_geos}")

                # Re-evaluate individually with proper cross-exclusion
                clusters = discover_geo_clusters(
                    filepath=filepath,
                    date_col=date_col,
                    geos=geos,
                    fixed_treatment=best_treatment_geos,
                    start_date=start_date,
                    end_date=end_date,
                    verbose=False
                )
        except Exception as e:
            if verbose:
                print(f"  ⚠ Failed to find clusters: {e}\n")
            scenarios.append({
                "pct_treatment": pct,
                "n_treatment": n_treat,
                "clusters": None,
                "duration": None,
                "error": str(e),
            })
            continue

        # 4b. Cross-validation per cluster
        validation = validate_geo_clusters(
            filepath=filepath,
            date_col=date_col,
            splits=clusters,
            treatment_start_date=end_date, # In DoE, end_date is the pre-post split for CV
            start_date=start_date,
            n_folds=n_folds,
            plot=False,
            verbose=False
        )
        cv_summary = validation["summary"]

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
            _print_scenario_table(clusters, duration, mde, cv_summary, total_geos=n_geos)

        scenarios.append({
            "pct_treatment": pct,
            "n_treatment": n_treat,
            "clusters": clusters,
            "duration": duration,
            "validation": cv_summary,
        })

    # 4. Final comparison table
    comparison = _build_comparison(scenarios, mde, experiment_days)

    if verbose:
        _print_comparison_table(comparison, mde)

    return {
        "scenarios": scenarios,
        "comparison": comparison,
    }


# ── DoE Helpers ───────────────────────────────────────────────────────────

def _print_scenario_table(clusters, duration, mde, cv_summary=None, total_geos=None):
    """Print compact per-cluster table for a scenario with CV metrics and donor pools."""
    cluster_results = duration["cluster_results"]
    consolidated = duration["consolidated"]

    is_auto = mde is None

    # ── Duration & MDE Grid ──
    if is_auto:
        print(f"\n  {'Cluster':<7} | {'Treatment':<10} | {'#Ctrl':<5} | {'σ':<8} | {'R²':<6} | {'MDE @21d':<9} | {'MDE @30d':<9} | {'MDE @60d':<9}")
        print("  " + "-" * 80)

        for i, (cl, cr) in enumerate(zip(clusters, cluster_results)):
            treat = ", ".join(cl["treatment"])
            if len(treat) > 10:
                treat = treat[:7] + "..."
            n_ctrl = len(cl["control"])
            sigma = cr["summary"]["sigma"]
            r2 = cr["summary"].get("r_squared", 0)
            curve = cr.get("mde_curve")

            mde_strs = []
            for d in [21, 30, 60]:
                if curve is not None:
                    val = curve.loc[curve["days"] == d, "mde"]
                    mde_strs.append(f"{val.values[0]*100:.2f}%" if len(val) > 0 else "N/A")
                else:
                    mde_strs.append("N/A")

            print(f"  {i:<7} | {treat:<10} | {n_ctrl:<5} | {sigma:<8.4f} | {r2:<6.4f} | {mde_strs[0]:<9} | {mde_strs[1]:<9} | {mde_strs[2]:<9}")

        # Consolidated row
        c_summary = consolidated["summary"]
        c_sigma = c_summary["sigma"]
        c_curve = consolidated.get("mde_curve")
        c_mdes = []
        for d in [21, 30, 60]:
            if c_curve is not None:
                val = c_curve.loc[c_curve["days"] == d, "mde"]
                c_mdes.append(f"{val.values[0]*100:.2f}%" if len(val) > 0 else "N/A")
            else:
                c_mdes.append("N/A")

        print("  " + "-" * 80)
        print(f"  {'CONSOL.':<7} | {'pooled':<10} | {'':<5} | {c_sigma:<8.4f} | {'':<6} | {c_mdes[0]:<9} | {c_mdes[1]:<9} | {c_mdes[2]:<9}")
    else:
        print(f"\n  {'Cluster':<7} | {'Treatment':<10} | {'#Ctrl':<5} | {'σ':<8} | {'R²':<6} | {'Min Days':<8} | {'Power':<7}")
        print("  " + "-" * 65)

        for i, (cl, cr) in enumerate(zip(clusters, cluster_results)):
            treat = ", ".join(cl["treatment"])
            if len(treat) > 10:
                treat = treat[:7] + "..."
            n_ctrl = len(cl["control"])
            sigma = cr["summary"]["sigma"]
            r2 = cr["summary"].get("r_squared", 0)
            best_days = cr["summary"].get("best_days")
            best_power = cr["summary"].get("best_power")

            days_str = f"{int(best_days)}d" if best_days else "N/A"
            power_str = f"{best_power:.1%}" if best_power else "N/A"

            print(f"  {i:<7} | {treat:<10} | {n_ctrl:<5} | {sigma:<8.4f} | {r2:<6.4f} | {days_str:<8} | {power_str:<7}")

        c_summary = consolidated["summary"]
        c_sigma = c_summary["sigma"]
        c_best = c_summary.get("best_days")
        c_power = c_summary.get("best_power")
        c_days_str = f"{int(c_best)}d" if c_best else "N/A"
        c_power_str = f"{c_power:.1%}" if c_power else "N/A"

        print("  " + "-" * 65)
        print(f"  {'CONSOL.':<7} | {'pooled':<10} | {'':<5} | {c_sigma:<8.4f} | {'':<6} | {c_days_str:<8} | {c_power_str:<7}")

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
    
    print("\n  CONTROL DESIGN (DONOR POOL & WEIGHTS):")
    for i, cl in enumerate(clusters):
        treat_name = ", ".join(cl["treatment"])
        weights = cl.get("control_weights", [])
        controls = cl["control"]
        
        # Sort donors by weight DESC
        donors = sorted(zip(controls, weights), key=lambda x: x[1], reverse=True)
        # Only show donors with weight > 0.001
        donor_strs = [f"{d} ({w:.3f})" for d, w in donors if w > 0.001]
        
        print(f"  Cluster {i} ({treat_name:<10}): ", end="")
        for j in range(0, len(donor_strs), 4):
            chunk = donor_strs[j:j+4]
            if j > 0: print(" " * 24, end="")
            print(" | ".join(chunk))

    # ── Cross-Validation Grid ──
    if cv_summary is not None and not cv_summary.empty:
        print(f"\n  {'Cluster':<7} | {'Treatment':<10} | {'R² Train':<8} | {'R² Test':<8} | {'MAPE Tr':<8} | {'MAPE Te':<8} | {'WAPE Tr':<8} | {'WAPE Te':<8}")
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

            # Flag overfitting
            gap = row['r2_train'] - row['r2_test']
            flag = " ⚠" if gap > 0.2 else ""

            print(f"  {i:<7} | {treat:<10} | {r2_tr:<8} | {r2_te:<8} | {mape_tr:<8} | {mape_te:<8} | {wape_tr:<8} | {wape_te:<8}{flag}")

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
            for d in [21, 30, 60]:
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


def _print_comparison_table(comparison_df, mde):
    """Print the final comparison table across scenarios."""
    if comparison_df.empty:
        print("\n  No valid scenarios found to compare.")
        return

    print("\n" + "=" * 85)
    print(" EXPERIMENT DESIGN COMPARISON ".center(85, "="))
    print("=" * 85)
    print("")
    
    is_auto = mde is None
    if is_auto:
        header = f"  {'Scenario':<8} | {'% Treated':<10} | {'Clusters':<8} | {'Distinct':<8} | {'σ Consol.':<10} | {'MDE @21d':<9} | {'MDE @30d':<9} | {'MDE @60d':<9}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in comparison_df.iterrows():
            print(f"  {int(row['Scenario']):<8} | {row['% Treated']:<10} | {int(row['Clusters']):<8} | {int(row['Distinct']):<8} | {row['sigma']:<10.4f} | {row['mde_21d']:<9} | {row['mde_30d']:<9} | {row['mde_60d']:<9}")
    else:
        header = f"  {'Scenario':<8} | {'% Treated':<10} | {'Clusters':<8} | {'Distinct':<8} | {'σ Consol.':<10} | {'Min Days':<10} | {'Power':<7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in comparison_df.iterrows():
            print(f"  {int(row['Scenario']):<8} | {row['% Treated']:<10} | {int(row['Clusters']):<8} | {int(row['Distinct']):<8} | {row['sigma']:<10.4f} | {row['best_days']:<10} | {row['best_power']:<7}")
    
    print("\n" + "=" * 85)

    print("\n" + "=" * 70)
