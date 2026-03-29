from ..geo.split import find_best_geo_clusters
from ..geo.validation import validate_geo_groups
from ..geo.duration import estimate_duration
from ..geo.synthetic import run_synthetic_control, plot_synthetic_control
from ..geo.placebo import run_placebo_tests, plot_placebo_tests

def run_geo_experiment(
    filepath,
    date_col,
    treatment_start_date,
    geos=None,
    n_treatment=1,
    fixed_treatment=None,
    mde=0.01,
    max_days=[21, 60],
    n_folds=1,
    random_state=None,
    plot=True,
    verbose=True
) -> dict:
    """
    Run a complete GeoLift experiment pipeline.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        treatment_start_date (str): Treatment start date isolating pre vs post.
        geos (list): List of candidate control geographies.
        n_treatment (int): Number of treatment groups.
        fixed_treatment (list): Hardcoded list of specific geos to treat.
        mde (float): Minimum detectable effect expected.
        max_days (int or list): Maximum days limit for the duration estimation, or a range [min, max].
        n_folds (int): Number of folds for Time Series Cross-Validation evaluating the Synthetic engine.
        random_state (int or np.random.Generator): Random state for placebo reproducibility.
        plot (bool): Whether to plot final graphical diagnostics.
        verbose (bool): Whether to print comprehensive terminal logging results.

    Returns:
        dict: Complete experiment diagnostic result, with subdictionaries by cluster.
    """
    # 1. Find best split and clusters
    clusters = find_best_geo_clusters(
        filepath=filepath,
        date_col=date_col,
        geos=geos,
        n_treatment=n_treatment,
        fixed_treatment=fixed_treatment,
        treatment_start_date=treatment_start_date,
        verbose=verbose
    )

    # 2. Results storage
    results = []

    # 3. Process each cluster
    for i, cluster in enumerate(clusters):
        if verbose:
            print(f"\n==================================================")
            print(f"=== CLUSTER {i} ===")
            print(f"==================================================")

        # 3.1 Validate current cluster
        validation = validate_geo_groups(
            filepath=filepath,
            date_col=date_col,
            splits=[cluster],
            treatment_start_date=treatment_start_date,
            n_folds=n_folds,
            cluster_idx=i,
            plot=plot,
            verbose=verbose
        )

        # 4. Estimate duration
        duration = estimate_duration(
            filepath=filepath,
            date_col=date_col,
            treatment_geo=cluster["treatment"], # Pass whole list
            control_geos=cluster["control"],
            treatment_start_date=treatment_start_date,
            mde=mde,
            max_days=max_days,
            cluster_idx=i,
            verbose=verbose
        )

        # 5. Run synthetic control
        synthetic = run_synthetic_control(
            filepath=filepath,
            date_col=date_col,
            treatment_geo=cluster["treatment"], # Pass whole list
            control_geos=cluster["control"],
            treatment_start_date=treatment_start_date,
            random_state=random_state,
            cluster_idx=i,
            plot=False,
            verbose=verbose
        )

        # 6. Placebo tests
        placebo = run_placebo_tests(
            filepath=filepath,
            date_col=date_col,
            control_geos=cluster["control"],
            treatment_start_date=treatment_start_date,
            observed_lift=synthetic["lift_mean_abs"],
            random_state=random_state,
            cluster_idx=i,
            plot=False,
            verbose=verbose
        )

        # 7. Plots (Show after all summaries)
        if plot:
            plot_synthetic_control(
                df=synthetic["df"],
                treatment_geo=cluster["treatment"], # Pass whole list
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
                placebo_lifts=placebo["placebo_lifts"],
                observed_lift=placebo["observed_lift"]
            )

        results.append({
            "cluster": cluster,
            "validation": validation,
            "duration": duration,
            "synthetic": synthetic,
            "placebo": placebo
        })

    if verbose and len(results) > 0:
        print("\n\n" + "="*50)
        print("=== GEO EXPERIMENT SUMMARY ===")
        print("="*50 + "\n")
        
        print("Objective:")
        print("Estimate incremental impact using GeoLift (synthetic control)\n")
        
        df_first = results[0]["synthetic"]["df"]
        idx_treat = results[0]["synthetic"]["plotting_data"]["treatment_idx"]
        
        pre_start = df_first[date_col].iloc[0].strftime('%b %Y')
        pre_end = df_first[date_col].iloc[idx_treat-1].strftime('%b %Y')
        
        post_start = df_first[date_col].iloc[idx_treat].strftime('%b %d')
        post_end = df_first[date_col].iloc[-1].strftime('%b %d')
        post_days = len(df_first) - idx_treat
        
        print(f"Pre-period: {pre_start} → {pre_end}")
        print(f"Intervention: {post_start} → {post_end} ({post_days} days)")
        
        n_clusters = len(results)
        num_treated = len(results[0]["cluster"]["treatment"])
        print(f"Clusters evaluated: {n_clusters} ({num_treated} geo per treatment)\n")
        print("-" * 50 + "\n")
        
        print("=== MODEL VALIDATION ===\n")
        print(f"{'Cluster':<7} | {'Corr':<5} | {'OOF R2':<6} | {'MAPE':<5} | {'MDE':<5} | {'Min Duration':<12} | {'Power':<6} | {'Robust'}")
        print("-" * 80)
        
        all_robust = True
        for i, res in enumerate(results):
            corr = res["duration"]["summary"]["correlation"]
            oof_r2 = res["validation"]["summary"].iloc[0]["r2_test"]
            mape_val = res["validation"]["summary"].iloc[0]["mape_test"]
            mde_val = res["duration"]["summary"]["mde"] * 100
            
            min_dur = res["duration"]["summary"]["best_days"]
            if min_dur:
                dur_str = f"{int(min_dur)}d"
                power_str = f"{res['duration']['summary']['best_power']*100:.2f}%"
            else:
                dur_str = "N/A"
                power_str = "N/A"
                
            rob = "✔" if oof_r2 > 0.6 else "⚠️"
            if oof_r2 <= 0.6: all_robust = False
                
            print(f"{i:<7} | {corr:<5.3f} | {oof_r2:<6.4f} | {mape_val:<5.4f} | {mde_val:<4.2f}% | {dur_str:<12} | {power_str:<6} | {rob}")
            
        print("")
        if all_robust:
            print("✔ Strong predictive performance across all clusters")
            print("✔ Stable cross-validation and low error")
            print("✔ No signs of overfitting\n")
        else:
            print("⚠️ Some clusters show lower predictive robustness. Review OOF R2 values carefully.\n")
            
        print("-" * 50 + "\n")
        
        print("=== CLUSTER-LEVEL INCREMENTAL IMPACT ===\n")
        print(f"{'Cluster':<7} | {'Treatment':<9} | {'Observed':<10} | {'Counterfactual':<14} | {'Lift (%)':<8} | {'Lift (abs)':<10} | {'CI (%)':<16} | {'CI (abs)':<16} | {'Sig'}")
        print("-" * 123)
        
        tot_lifts_pct = []
        tot_lifts_abs = []
        ci_lowers_pct = []
        ci_uppers_pct = []
        ci_lowers_abs = []
        ci_uppers_abs = []
        tot_real_abs = []
        tot_synth_abs = []
        
        for i, res in enumerate(results):
            treatment_str = ", ".join(res["cluster"]["treatment"])
            if len(treatment_str) > 9:
                treatment_str = treatment_str[:6] + "..."
                
            syn = res["synthetic"]
            lift_pct = syn["lift_total"] / syn["plotting_data"]["post_synth"].sum()
            lift_abs = syn["lift_total"]
            
            ci_l_pct = syn["bootstrap"]["ci_lower_total_pct"]
            ci_u_pct = syn["bootstrap"]["ci_upper_total_pct"]
            
            ci_l_abs = syn["bootstrap"]["ci_lower_total_abs"]
            ci_u_abs = syn["bootstrap"]["ci_upper_total_abs"]
            
            sig = "✔" if (ci_l_abs > 0 or ci_u_abs < 0) else "⚠️"
            
            ci_str = f"[{ci_l_pct*100:.2f}, {ci_u_pct*100:.2f}]"
            ci_abs_str = f"[{ci_l_abs:.2f}, {ci_u_abs:.2f}]"
            
            post_synth_sum = syn["plotting_data"]["post_synth"].sum()
            post_real_sum = syn["plotting_data"]["post_real"].sum()
            
            print(f"{i:<7} | {treatment_str:<9} | {post_real_sum:<10.2f} | {post_synth_sum:<14.2f} | {lift_pct*100:<7.2f}% | {lift_abs:<10.2f} | {ci_str:<16} | {ci_abs_str:<16} | {sig}")
            
            tot_lifts_pct.append(lift_pct)
            tot_lifts_abs.append(lift_abs)
            ci_lowers_pct.append(ci_l_pct)
            ci_uppers_pct.append(ci_u_pct)
            ci_lowers_abs.append(ci_l_abs)
            ci_uppers_abs.append(ci_u_abs)
            tot_real_abs.append(post_real_sum)
            tot_synth_abs.append(post_synth_sum)
            
        print("\n" + "-" * 50 + "\n")
        
        print("=== INCREMENTAL IMPACT ===\n")
        
        sum_real = sum(tot_real_abs)
        sum_synth = sum(tot_synth_abs)
        sum_lift = sum_real - sum_synth
        agg_lift_pct = sum_lift / sum_synth if sum_synth != 0 else 0.0
        
        agg_ci_l_abs = sum(ci_lowers_abs)
        agg_ci_u_abs = sum(ci_uppers_abs)
        agg_ci_l_pct = agg_ci_l_abs / sum_synth if sum_synth != 0 else 0.0
        agg_ci_u_pct = agg_ci_u_abs / sum_synth if sum_synth != 0 else 0.0
        
        print(f"Total Observed: {sum_real:.2f}")
        print(f"Total Synthetic / Counterfactual: {sum_synth:.2f}")
        
        print(f"\nIncremental Lift (abs): {sum_lift:.2f}")
        print(f"Incremental Lift CI (abs): [{agg_ci_l_abs:.2f}, {agg_ci_u_abs:.2f}]")
        
        print(f"\nIncremental Lift (%): {agg_lift_pct*100:.2f}%")
        print(f"Incremental CI (%): [{agg_ci_l_pct*100:.2f}%, {agg_ci_u_pct*100:.2f}%]\n")

    return {
        "summary": {
            "clusters": clusters
        },
        "results": results
    }

def run_geo_requirements(
    filepath,
    date_col,
    treatment_start_date=None,
    n_treatment=1,
    fixed_treatment=None,
    n_folds=5,
    mde=0.015,
    max_days=[21, 60],
    verbose=True
) -> dict:
    """
    Consolidate GEO experiment requirements: finds clusters, validates them, and estimates duration.
    Produces a structured report for each cluster found.
    """
    # 1. Find best clusters
    clusters = find_best_geo_clusters(
        filepath=filepath,
        date_col=date_col,
        n_treatment=n_treatment,
        fixed_treatment=fixed_treatment,
        treatment_start_date=treatment_start_date,
        verbose=verbose
    )

    # 2. Validate clusters (silent, we'll format the output manually)
    validation_results = validate_geo_groups(
        filepath=filepath,
        date_col=date_col,
        splits=clusters,
        treatment_start_date=treatment_start_date,
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
            treatment_start_date=treatment_start_date,
            mde=mde,
            max_days=max_days,
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
            print(f"Std Residual: {stats['std_residual_regression']:.4f}")
            print(f"R-Squared: {stats['r_squared']:.4f}")
            print(f"Correlation: {stats['correlation']:.3f}")

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
        
        final_results.append({
            "cluster": cluster,
            "validation": cv_metric.to_dict(),
            "duration": stats
        })

    if verbose:
        print("\n=== GEO EXPERIMENT REQUIREMENT SUMMARY ===\n")
        print(f"{'Cluster':<7} | {'Treatment':<12} | {'Correlation':<11} | {'OOF R2':<6} | {'MAPE':<6} | {'MDE':<5} | {'Min Duration':<12} | {'Power':<5}")
        print("-" * 80)
        
        max_duration = 0
        all_durations_found = True
        
        for i, res in enumerate(final_results):
            treatment_str = ", ".join(res["cluster"]["treatment"])
            if len(treatment_str) > 12:
                treatment_str = treatment_str[:9] + "..."
            
            corr = res["duration"]["correlation"]
            oof_r2 = res["validation"]["r2_test"]
            mape_val = res["validation"]["mape_test"]
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
        
        all_treatments = set()
        all_controls = set()
        for res in final_results:
            all_treatments.update(res["cluster"]["treatment"])
            all_controls.update(res["cluster"]["control"])
        
        print(f"Selected Clusters For Treatment: {', '.join(sorted(list(all_treatments)))}")
        print(f"Selected Clusters For Control: {', '.join(sorted(list(all_controls)))}\n")
        
        if all_durations_found and max_duration > 0:
            print(f"Recommended duration for intervention: {max_duration} days")
            print("\nRationale:")
            print("Duration chosen to satisfy statistical power requirements across all clusters.")
            print("Ensures consistent measurement and comparability.")
        else:
            print("Recommended duration for intervention: N/A")
            print("\nRationale:")
            print("One or more clusters did not reach the minimum statistical power within the maximum tested days.")
            print("Consider increasing the MDE, the max_days limit, or relaxing the fixed geographical splits.")
        print("\n")

    return {
        "clusters": clusters,
        "results": final_results
    }