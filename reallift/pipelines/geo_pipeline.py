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
    max_days=60,
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
        treatment_start_date (str): Treatment start date.
        geos (list): List of geographies.
        n_treatment (int): Number of treatment groups.
        fixed_treatment (list): Fixed treatment groups.
        mde (float): Minimum detectable effect.
        max_days (int): Maximum days.
        n_folds (int): Number of folds for CV in validation.
        random_state (int or np.random.Generator): Random state for reproducibility.
        plot (bool): Whether to plot.
        verbose (bool): Whether to print results.

    Returns:
        dict: Complete experiment result.
    """
    # 1. Find best split and clusters
    clusters = find_best_geo_clusters(
        filepath=filepath,
        date_col=date_col,
        geos=geos,
        n_treatment=n_treatment,
        fixed_treatment=fixed_treatment,
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

    return {
        "summary": {
            "clusters": clusters
        },
        "results": results
    }

def run_geo_requirements(
    filepath,
    date_col,
    n_treatment=1,
    fixed_treatment=None,
    n_folds=5,
    mde=0.015,
    max_days=60,
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
        verbose=verbose
    )

    # 2. Validate clusters (silent, we'll format the output manually)
    validation_results = validate_geo_groups(
        filepath=filepath,
        date_col=date_col,
        splits=clusters,
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
            mde=mde,
            max_days=max_days,
            verbose=False # Silent
        )
        stats = duration_results["summary"]

        # 4. Extract CV metrics for this specific cluster
        cv_metric = cv_summary.iloc[i]

        if verbose:
            print(f"\n==================================================")
            print(f"=== CLUSTER {i} ===")
            print(f"==================================================")
            print(f"\nTreatment: {cluster['treatment']}")
            print(f"Control: {cluster['control']}")

            print(f"\n--- CROSS-VALIDATION ({n_folds} FOLDS) ---")
            print(f"Train R2: {cv_metric['r2_train']:.4f} | OOF R2: {cv_metric['r2_test']:.4f}")
            print(f"Train MAPE: {cv_metric['mape_train']:.4f} | OOF MAPE: {cv_metric['mape_test']:.4f}")
            print(f"Train WAPE: {cv_metric['wape_train']:.4f} | OOF WAPE: {cv_metric['wape_test']:.4f}")

            print(f"\n--- STATISTICS ---")
            print(f"Mean: {stats['mean']:.2f}")
            print(f"Std naive: {stats['std_naive']:.2f}")
            print(f"Std residual (mean): {stats['std_residual_simple']:.2f}")
            print(f"Std residual (regression): {stats['std_residual_regression']:.4f}")
            print(f"Correlation: {stats['correlation']:.3f}")

            print(f"\n--- DETECTABLE EFFECT ---")
            print(f"MDE: {mde*100:.2f}% | Abs: {stats['delta_abs']:.2f}")

            print(f"\n--- EXPERIMENT REQUIREMENTS ---")
            if stats['best_days']:
                print(f"Min duration: {int(stats['best_days'])} days")
                print(f"Expected power: {stats['best_power']:.2%}")
            else:
                print(f"⚠️ Min power target not reached in {max_days} days")
                print(f"👉 Estimated duration needed: {stats['estimated_days_needed']} days")
            
            print("") # Extra newline for spacing
        
        final_results.append({
            "cluster": cluster,
            "validation": cv_metric.to_dict(),
            "duration": stats
        })

    return {
        "clusters": clusters,
        "results": final_results
    }