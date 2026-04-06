import pandas as pd
import numpy as np
from itertools import combinations
from math import comb as math_comb
import cvxpy as cp
from sklearn.linear_model import ElasticNet
from reallift.utils.preprocessing import log_diff_transform, scale_data

def discover_geo_clusters(
    filepath,
    date_col,
    geos=None,
    n_treatment=3,
    fixed_treatment=None,
    start_date=None,
    end_date=None,
    use_elasticnet=True,
    search_mode="auto",
    verbose=True,
    show_results=True
) -> list:
    """
    Find the best geo split and build clusters for treatment and control groups.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        geos (list): List of candidate control geographies.
        n_treatment (int): Number of treatment groups to simulate if fixed is None.
        fixed_treatment (list): Hardcoded list of specific geos to treat.
        start_date (str): YYYY-MM-DD date when history begins (optional).
        end_date (str): YYYY-MM-DD date when history ends (optional).
        use_elasticnet (bool): Whether to use ElasticNet to pre-filter controls.
        search_mode (str): Search strategy for treatment selection.
            - "exhaustive": test all C(n,k) combinations (default for small problems).
            - "ranking": screen geos individually, pick top-k, re-evaluate.
            - "auto": use ranking if C(n,k) > 1000, exhaustive otherwise.
        verbose (bool): Whether to print running logs.

    Returns:
        list: List of dictionaries containing the best cluster splits.
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col])

    if start_date is not None:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.to_datetime(end_date)]
    
    df = df.groupby(date_col).sum(numeric_only=True).reset_index()
    df = df.sort_values(date_col).reset_index(drop=True)

    if geos is None:
        geos = [col for col in df.columns if col != date_col]

    alpha_grid = [0.001, 0.01, 0.1] if use_elasticnet else [0.0]
    l1_grid = [0.2, 0.5, 0.8] if use_elasticnet else [0.0]

    # ── Determine effective search mode ───────────────────────────────────
    if fixed_treatment is not None:
        effective_mode = "fixed"
    elif n_treatment == 1:
        effective_mode = "exhaustive"
    else:
        n_combos = math_comb(len(geos), n_treatment)
        if search_mode == "exhaustive":
            effective_mode = "exhaustive"
        elif search_mode == "ranking":
            effective_mode = "ranking"
        else:  # auto
            effective_mode = "ranking" if n_combos > 1000 else "exhaustive"

    # ── RANKING MODE: two-phase approach ──────────────────────────────────
    if effective_mode == "ranking":
        if verbose:
            n_combos = math_comb(len(geos), n_treatment)
            print(f"\nRanking mode: {n_combos:,} combinations → {len(geos)} + {n_treatment} evaluations")

        # Phase 1: Screen all geos individually (no cross-exclusion)
        phase1_combos = [[g] for g in geos]
        phase1_results = _evaluate_combinations(
            df, geos, phase1_combos,
            all_treatment_geos=set(),
            use_elasticnet=use_elasticnet,
            alpha_grid=alpha_grid, l1_grid=l1_grid,
            verbose=verbose, desc="Screening geos"
        )

        # Sort and pick top-k
        phase1_results.sort(key=lambda x: x["synthetic_error_ratio"])
        top_k = [r["treatment"][0] for r in phase1_results[:n_treatment]]

        if verbose:
            print(f"Top {n_treatment} candidates: {top_k}")
            print("Re-evaluating with proper control exclusions...\n")

        # Phase 2: Re-evaluate with cross-exclusion
        phase2_combos = [[t] for t in top_k]
        results = _evaluate_combinations(
            df, geos, phase2_combos,
            all_treatment_geos=set(top_k),
            use_elasticnet=use_elasticnet,
            alpha_grid=alpha_grid, l1_grid=l1_grid,
            verbose=False, desc=None
        )

        clusters = _build_clusters(results, use_elasticnet)
        if verbose and show_results:
            _print_results(clusters, is_fixed=True)
        return clusters

    # ── FIXED / EXHAUSTIVE MODE ───────────────────────────────────────────
    all_treatment_geos = set(fixed_treatment) if fixed_treatment is not None else set()

    if fixed_treatment is not None:
        treatment_combinations = [[t] for t in fixed_treatment]
    else:
        treatment_combinations = list(combinations(geos, n_treatment))

    results = _evaluate_combinations(
        df, geos, treatment_combinations,
        all_treatment_geos=all_treatment_geos,
        use_elasticnet=use_elasticnet,
        alpha_grid=alpha_grid, l1_grid=l1_grid,
        verbose=verbose,
        desc="Evaluating Combinations" if fixed_treatment is None else "Evaluating Fixed"
    )

    if use_elasticnet:
        results = sorted(results, key=lambda x: x["synthetic_error_ratio"])
    else:
        results = sorted(results, key=lambda x: x["rmspe"])

    clusters = _build_clusters(results, use_elasticnet)

    if verbose and show_results:
        _print_results(clusters, is_fixed=(fixed_treatment is not None))

    return clusters if fixed_treatment else clusters[:5]


# ── Internal helpers ──────────────────────────────────────────────────────

def _evaluate_combinations(df, geos, treatment_combinations, all_treatment_geos,
                           use_elasticnet, alpha_grid, l1_grid, verbose, desc):
    """Run the evaluation loop over treatment combinations."""
    if verbose and desc:
        try:
            from tqdm import tqdm
            iterator = tqdm(treatment_combinations, desc=desc, leave=True)
        except ImportError:
            iterator = treatment_combinations
            print(f"{desc}: {len(treatment_combinations)} combinations...")
    else:
        iterator = treatment_combinations

    results = []

    for treatment_comb in iterator:
        treatment_comb = list(treatment_comb)
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
                    if use_elasticnet:
                        model = ElasticNet(alpha=a, l1_ratio=l1, max_iter=10000)
                        model.fit(X_scaled, y)

                        coefs = model.coef_
                        selected_controls = [
                            control_pool[i]
                            for i in range(len(control_pool))
                            if coefs[i] > 0
                        ]

                        if len(selected_controls) == 0:
                            idx_max = np.argmax(np.abs(coefs))
                            selected_controls = [control_pool[idx_max]]
                    else:
                        selected_controls = control_pool

                    # Synthetic Control optimization
                    final_weights = []
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
                        final_controls_with_w = [(c, w_val) for c, w_val in zip(selected_controls, w_vals) if w_val > 0.001]
                        
                        if len(final_controls_with_w) == 0:
                            final_controls = selected_controls
                            final_weights = [1.0 / len(selected_controls)] * len(selected_controls)
                        else:
                            final_controls = [x[0] for x in final_controls_with_w]
                            final_weights = [x[1] for x in final_controls_with_w]
                            
                            sum_w = sum(final_weights)
                            if sum_w > 0:
                                final_weights = [w / sum_w for w in final_weights]
                    except Exception:
                        final_controls = selected_controls
                        final_weights = [1.0 / len(selected_controls)] * len(selected_controls)

                    X_selected = df_transformed[final_controls].values

                    if use_elasticnet:
                        X_selected_scaled, _ = scale_data(X_selected)
                        model.fit(X_selected_scaled, y)
                        y_pred = model.predict(X_selected_scaled)
                    else:
                        weight_array = np.array(final_weights)
                        y_pred = X_selected @ weight_array
                        
                    residual = y - y_pred

                    std_residual = float(np.std(residual))
                    rmspe = float(np.sqrt(np.mean(residual**2)))
                    
                    if np.std(y) > 0 and np.std(y_pred) > 0:
                        corr = float(np.corrcoef(y, y_pred)[0, 1])
                    else:
                        corr = 0.0

                    candidate = {
                        "treatment": list(treatment_comb),
                        "control": final_controls,
                        "control_weights": final_weights,
                        "std_residual": std_residual,
                        "rmspe": rmspe,
                        "correlation": corr,
                        "synthetic_error_ratio": std_residual / (corr + 1e-6),
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

    return results


def _build_clusters(results, use_elasticnet):
    """Sort results and build cluster list."""
    if use_elasticnet:
        results = sorted(results, key=lambda x: x["synthetic_error_ratio"])
    else:
        results = sorted(results, key=lambda x: x["rmspe"])

    clusters = []
    for res in results:
        clusters.append({
            "treatment": res["treatment"],
            "control": res["control"],
            "control_weights": res["control_weights"],
            "correlation": res["correlation"],
            "std_residual": res["std_residual"],
            "rmspe": res["rmspe"],
            "synthetic_error_ratio": res["synthetic_error_ratio"],
            "n_controls": res["n_controls"],
            "alpha": res["alpha"],
            "l1_ratio": res["l1_ratio"]
        })
    return clusters


def _print_results(clusters, is_fixed):
    """Print final results in formatted output."""
    from IPython.display import display
    
    if not is_fixed:
        print("\n\n" + "-"*60)
        print("SEARCH ENGINE COMPLETED")
        print(f"Top {min(len(clusters), 5)} experiment design recommendations (ranked by lowest numerical risk):")
        print("Tip: Recommendation 0 is statistically your most precise choice for simulation.")
        print("     It minimizes the 'Synthetic Error Ratio' (Std Residual / Correlation), balancing low error with high synchronization.")
        print("-"*60)
        label_prefix = "RECOMMENDATION"
    else:
        print("\n\n" + "-"*60)
        print("FIXED TREATMENT ANALYSIS COMPLETED")
        print("Evaluating exclusive Synthetic Controls for the mandatory treatment units.")
        print("-"*60)
        label_prefix = "TEST CLUSTER"

    print("\n" + "="*60)
    print(" FINAL RESULTS (DONOR POOL & WEIGHTS) ".center(60, "="))
    print("="*60)
    
    display_clusters = clusters if is_fixed else clusters[:5]
    
    for i, c in enumerate(display_clusters):
        treatment_str = ", ".join(c['treatment'])
        print(f"\n{label_prefix} {i} | Treatment: [{treatment_str}] | Correlation: {c['correlation']:.4f} | RMSPE: {c['rmspe']:.4f} | Std Residual: {c['std_residual']:.4f} | Synthetic Error Ratio: {c['synthetic_error_ratio']:.4f}")
        print("-" * 60)
        
        controls_with_weights = list(zip(c['control'], c.get('control_weights', [])))
        sorted_controls = sorted(controls_with_weights, key=lambda x: x[1], reverse=True)
        donor_strings = [f"{geo} ({w:.3f})" for geo, w in sorted_controls]
        
        print("Donor Pool:")
        for j in range(0, len(donor_strings), 4):
            chunk = donor_strings[j:j+4]
            row_str = " | ".join(f"{item:<16}" for item in chunk)
            print("  " + row_str)
            
    print("\n" + "="*60)