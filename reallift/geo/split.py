import pandas as pd
import numpy as np
from itertools import combinations
import cvxpy as cp
from sklearn.linear_model import ElasticNet
from reallift.utils.preprocessing import log_diff_transform, scale_data

def find_best_geo_clusters(
    filepath,
    date_col,
    geos=None,
    n_treatment=3,
    fixed_treatment=None,
    treatment_start_date=None,
    verbose=True
) -> list:
    """
    Find the best geo split and build clusters for treatment and control groups.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        geos (list): List of candidate control geographies.
        n_treatment (int): Number of treatment groups to simulate if fixed is None.
        fixed_treatment (list): Hardcoded list of specific geos to treat.
        treatment_start_date (str): YYYY-MM-DD date when treatment begins. Pre-treatment runs until this date.
        verbose (bool): Whether to print running logs.

    Returns:
        list: List of dictionaries containing the best cluster splits.
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col])

    if treatment_start_date is not None:
        df = df[df[date_col] < pd.to_datetime(treatment_start_date)]
    
    # Aggregate by date to handle duplicate entries
    # Ensures one observation per date before running the splitting algorithm
    df = df.groupby(date_col).sum(numeric_only=True).reset_index()
    df = df.sort_values(date_col).reset_index(drop=True)

    if geos is None:
        geos = [col for col in df.columns if col != date_col]

    results = []

    alpha_grid = [0.001, 0.01, 0.1]
    l1_grid = [0.2, 0.5, 0.8]

    # Global set of geos that are part of ANY treatment to exclude from ALL control pools
    all_treatment_geos = set(fixed_treatment) if fixed_treatment is not None else set()

    if fixed_treatment is not None:
        # If we have multiple fixed geos, we evaluate each one individually 
        # to find its own best controls, avoiding identical metrics.
        treatment_combinations = [[t] for t in fixed_treatment]
    else:
        treatment_combinations = combinations(geos, n_treatment)

    for treatment_comb in treatment_combinations:
        # Exclude ALL treatment geos from the control pool, not just the current combination
        forbidden_geos = set(treatment_comb).union(all_treatment_geos)
        control_pool = [g for g in geos if g not in forbidden_geos]

        try:
            df_transformed = log_diff_transform(df, treatment_comb + control_pool)

            y = df_transformed[treatment_comb].mean(axis=1).values
            X = df_transformed[control_pool].values

            X_scaled, _ = scale_data(X)

            best_local = None

            for a in alpha_grid:
                for l1 in l1_grid:
                    model = ElasticNet(alpha=a, l1_ratio=l1, max_iter=10000)
                    model.fit(X_scaled, y)

                    coefs = model.coef_
                    # Filter controls with non-negative coefficients (aligns better with synthetic control)
                    selected_controls = [
                        control_pool[i]
                        for i in range(len(control_pool))
                        if coefs[i] > 0
                    ]

                    if len(selected_controls) == 0:
                        # Fallback to absolute value if everything is zero/negative
                        idx_max = np.argmax(np.abs(coefs))
                        selected_controls = [control_pool[idx_max]]

                    # Prune zero-weight controls using True Synthetic Optimization
                    try:
                        X_syn = df[selected_controls].values.astype(float)
                        y_syn = df[list(treatment_comb)].mean(axis=1).values.astype(float)
                        
                        y_mean_syn = y_syn.mean()
                        if y_mean_syn == 0: y_mean_syn = 1e-10
                        X_mean_syn = X_syn.mean(axis=0)
                        X_mean_syn[X_mean_syn == 0] = 1e-10

                        y_norm_syn = y_syn / y_mean_syn
                        X_norm_syn = X_syn / X_mean_syn

                        w_syn = cp.Variable(len(selected_controls))
                        alpha_syn = cp.Variable()

                        obj_syn = cp.Minimize(cp.sum_squares(y_norm_syn - (X_norm_syn @ w_syn + alpha_syn)))
                        cons_syn = [w_syn >= 0, cp.sum(w_syn) == 1]
                        prob_syn = cp.Problem(obj_syn, cons_syn)
                        prob_syn.solve(solver=cp.SCS, verbose=False)

                        w_vals = np.array(w_syn.value).flatten()
                        final_controls = [c for c, w_val in zip(selected_controls, w_vals) if w_val > 0.001]
                        if len(final_controls) == 0:
                            final_controls = selected_controls
                    except Exception:
                        final_controls = selected_controls

                    X_selected = df_transformed[final_controls].values
                    X_selected_scaled, _ = scale_data(X_selected)

                    # Re-fit only on selected controls to get the final score
                    model.fit(X_selected_scaled, y)
                    y_pred = model.predict(X_selected_scaled)
                    residual = y - y_pred

                    std_residual = float(np.std(residual))
                    
                    # Handle zero variance to avoid RuntimeWarnings in corrcoef
                    if np.std(y) > 0 and np.std(y_pred) > 0:
                        corr = float(np.corrcoef(y, y_pred)[0, 1])
                    else:
                        corr = 0.0

                    candidate = {
                        "treatment": list(treatment_comb),
                        "control": final_controls,
                        "std_residual": std_residual,
                        "correlation": corr,
                        "n_controls": len(selected_controls),
                        "alpha": a,
                        "l1_ratio": l1
                    }

                    if best_local is None or std_residual < best_local["std_residual"]:
                        best_local = candidate

            if best_local:
                results.append(best_local)

        except Exception as e:
            if verbose:
                print(f"Error with combination {treatment_comb}: {e}")
            continue

    if len(results) == 0:
        raise ValueError("No valid combinations found.")

    # Sort results to find the best overall
    results = sorted(results, key=lambda x: x["std_residual"] / (x["correlation"] + 1e-6))
    
    # NEW: Return all evaluated combinations as clusters
    # This ensures that each fixed treatment gets its own cluster with unique metrics.
    clusters = []
    for i, res in enumerate(results):
        clusters.append({
            "treatment": res["treatment"], # Keep full treatment list
            "control": res["control"],
            "correlation": res["correlation"],
            "std_residual": res["std_residual"],
            "n_controls": res["n_controls"],
            "alpha": res["alpha"],
            "l1_ratio": res["l1_ratio"]
        })

    if verbose:
        print("\n=== BEST CLUSTERS FOUND ===")
        # Limit to top 5 if not using fixed_treatment to avoid spam
        display_clusters = clusters if fixed_treatment else clusters[:5]
        for i, c in enumerate(display_clusters):
            print(f"Cluster {i}: Treatment {c['treatment']}, Correlation {c['correlation']:.4f}")

    return clusters if fixed_treatment else clusters[:5]