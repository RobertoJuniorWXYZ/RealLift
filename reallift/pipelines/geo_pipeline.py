from ..geo.split import find_best_geo_split, build_geo_clusters
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
    mde=0.1,
    max_days=60,
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
        random_state (int or np.random.Generator): Random state for reproducibility.
        plot (bool): Whether to plot.
        verbose (bool): Whether to print results.

    Returns:
        dict: Complete experiment result.
    """
    # 1. Find best split
    splits = find_best_geo_split(
        filepath=filepath,
        date_col=date_col,
        geos=geos,
        n_treatment=n_treatment,
        fixed_treatment=fixed_treatment,
        verbose=verbose
    )

    # 2. Build clusters
    clusters = build_geo_clusters(splits)

    # 3. Validate groups
    validation = validate_geo_groups(
        filepath=filepath,
        date_col=date_col,
        splits=splits,
        treatment_start_date=treatment_start_date,
        plot=plot,
        verbose=verbose
    )

    results = []

    for cluster in clusters:
        # 4. Estimate duration
        duration = estimate_duration(
            filepath=filepath,
            date_col=date_col,
            treatment_geo=cluster["treatment"][0],
            control_geos=cluster["control"],
            mde=mde,
            max_days=max_days,
            verbose=verbose
        )

        # 5. Run synthetic control
        synthetic = run_synthetic_control(
            filepath=filepath,
            date_col=date_col,
            treatment_geo=cluster["treatment"][0],
            control_geos=cluster["control"],
            treatment_start_date=treatment_start_date,
            random_state=random_state,
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
            plot=False,
            verbose=verbose
        )

        # 7. Plots (Show after both summaries)
        if plot:
            plot_synthetic_control(
                df=synthetic["df"],
                treatment_geo=cluster["treatment"][0],
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
            "splits": splits,
            "clusters": clusters
        },
        "results": results
    }