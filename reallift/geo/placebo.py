import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .synthetic import run_synthetic_control

def run_placebo_tests(
    filepath,
    date_col,
    control_geos,
    treatment_start_date,
    observed_lift,
    n_placebos=10,
    random_state=None,
    plot=True,
    verbose=True
) -> dict:
    """
    Run placebo tests by treating control geographies as treated.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        control_geos (list): Control geographies.
        treatment_start_date (str): Treatment start date.
        observed_lift (float): The actual observed lift to compare against.
        n_placebos (int): Number of placebo tests.
        random_state (int or np.random.Generator): Random state for reproducibility.
        plot (bool): Whether to plot the placebo distribution.
        verbose (bool): Whether to print results.

    Returns:
        dict: Placebo test results.
    """
    placebo_lifts = []

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
                random_state=random_state,
                plot=False,
                verbose=False
            )
            placebo_lifts.append(result["lift_mean_abs"])
        except:
            continue

    p_value = np.mean(np.abs(placebo_lifts) >= np.abs(observed_lift))

    if verbose:
        print("\n=== PLACEBO TESTS SUMMARY ===")
        print(f"Number of placebo tests: {len(placebo_lifts)}")
        print(f"Average placebo lift (noise): {np.mean(placebo_lifts):.4f}")
        print(f"Observed treatment lift: {observed_lift:.4f}")
        print(f"P-value (placebo): {p_value:.4f}")
        
        if p_value < 0.1:
            direction = "higher" if observed_lift > 0 else "lower"
            print(f"✔ High confidence: Observed lift is significantly {direction} than random placebos.")
        else:
            print("✖ Warning: Observed lift is within the range of random noise (high p-value).")

    if plot and len(placebo_lifts) > 0:
        plot_placebo_tests(placebo_lifts, observed_lift)

    return {
        "placebo_lifts": placebo_lifts,
        "observed_lift": observed_lift,
        "p_value": p_value
    }

def plot_placebo_tests(placebo_lifts, observed_lift):
    """
    Generate plot for placebo tests.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(placebo_lifts, color='skyblue', edgecolor='black', alpha=0.7, label='Placebo Lifts (Noise)')
    plt.axvline(observed_lift, color='red', linestyle='--', linewidth=2, label=f'Observed Lift ({observed_lift:.2f})')
    
    # Add a symmetrical line for two-sided visualization if lift is significant
    if observed_lift != 0:
        plt.axvline(-observed_lift, color='orange', linestyle=':', linewidth=1, label='Symmetrical Threshold')
        
    plt.title('Placebo Tests Distribution (Two-sided)')
    plt.xlabel('Estimated Lift (Absolute)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()