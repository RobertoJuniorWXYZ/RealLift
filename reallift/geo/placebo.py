import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .synthetic import run_synthetic_control

def run_placebo_tests(
    filepath,
    date_col,
    control_geos,
    treatment_start_date,
    observed_pre_mspe,
    observed_post_mspe,
    observed_lift=None,
    treatment_end_date=None,
    start_date=None,
    end_date=None,
    n_placebos=10,
    random_state=None,
    cluster_idx=None,
    plot=True,
    verbose=True
) -> dict:
    """
    Run placebo tests comparing the MSPE Ratio (Post/Pre) of the observed
    treatment against those of control geographies treated as placebos.

    Methodology: Abadie et al. (2010), comparing the distribution of the 
    ratio of post-intervention MSPE to pre-intervention MSPE.
    """
    placebo_ratios = []
    
    # Calculate Observed Ratio
    observed_ratio = observed_post_mspe / observed_pre_mspe if observed_pre_mspe > 0 else 0

    # Execute Placebo Runs
    for i in range(min(n_placebos, len(control_geos))):
        placebo_geo = control_geos[i]
        placebo_controls = [g for g in control_geos if g != placebo_geo]

        try:
            result = run_synthetic_control(
                filepath=filepath,
                date_col=date_col,
                treatment_geo=placebo_geo,
                control_geos=placebo_controls,
                treatment_start_date=treatment_start_date,
                treatment_end_date=treatment_end_date,
                start_date=start_date,
                end_date=end_date,
                random_state=random_state,
                plot=False,
                verbose=False
            )
            # Use MSPE Ratio: Post-MSPE / Pre-MSPE
            p_ratio = result["post_mspe"] / result["pre_mspe"] if result["pre_mspe"] > 0 else 0
            placebo_ratios.append(p_ratio)
        except:
            continue

    if len(placebo_ratios) == 0:
        p_value = 1.0
    else:
        # P-value is the proportion of placebo ratios greater than or equal to the observed ratio
        p_value = np.mean(np.array(placebo_ratios) >= observed_ratio)

    if verbose:
        header = "=== GEO PLACEBO TESTS ==="
        if cluster_idx is not None:
            header = f"=== GEO PLACEBO TESTS (Cluster {cluster_idx}) ==="
        print(f"\n{header}")
        print(f"Number of placebo tests: {len(placebo_ratios)}")
        print(f"Observed MSPE Ratio: {observed_ratio:.4f}")
        print(f"P-value (placebo): {p_value:.4f}")
        
        if p_value <= 0.10:
            print(f"✔ High confidence: Observed deviation is significantly higher than random noise (MSPE Ratio).")
        else:
            print("✖ Warning: Observed deviation is within the range of random noise (high p-value).")

    if plot and len(placebo_ratios) > 0:
        plot_placebo_tests(placebo_ratios, observed_ratio)

    return {
        "placebo_ratios": placebo_ratios,
        "observed_ratio": observed_ratio,
        "p_value": p_value
    }

def plot_placebo_tests(placebo_ratios, observed_ratio):
    """
    Generate plot for placebo tests using MSPE Ratio.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(placebo_ratios, color='skyblue', edgecolor='black', alpha=0.7, label='Placebo MSPE Ratios')
    plt.axvline(observed_ratio, color='red', linestyle='--', linewidth=2, label=f'Observed Ratio ({observed_ratio:.2f})')
    
    plt.title('Placebo Tests Distribution (MSPE Ratio: Post/Pre)')
    plt.xlabel('MSPE Ratio (Post-MSPE / Pre-MSPE)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()