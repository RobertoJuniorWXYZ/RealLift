import pandas as pd
import numpy as np
from ..geo.discovery import discover_geo_clusters
from ..geo.validation import validate_geo_clusters
from ..geo.duration import estimate_duration
from ..geo.synthetic import run_synthetic_control, plot_synthetic_control
from ..geo.did import run_matched_did, plot_matched_did
from ..geo.placebo import run_placebo_tests, plot_placebo_tests
from .reporting import print_experiment_summary

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
            treatment_start_date=treatment_start_date,
            start_date=start_date,
            end_date=treatment_start_date,
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
        print_experiment_summary(results, date_col, experiment_type=experiment_type, random_state=random_state)

    return {
        "summary": {
            "clusters": clusters
        },
        "results": results
    }
