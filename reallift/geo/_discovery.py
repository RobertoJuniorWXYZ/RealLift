import pandas as pd
import numpy as np
from itertools import combinations
from math import comb as math_comb
import cvxpy as cp
from sklearn.linear_model import ElasticNet
from reallift.utils.preprocessing import log_diff_transform, scale_data

def discover_geo_clusters(
    filepath=None,
    date_col=None,
    geos=None,
    n_treatment=3,
    fixed_treatment=None,
    start_date=None,
    end_date=None,
    method="penalized_scm",
    search_mode="auto",
    alpha=0.01,
    l1_ratio=0.5,
    n_jobs=None,
    verbose=True,
    show_results=True,
    check_oof=True,
    df=None
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
        method (str): Donor weighting strategy — "penalized_scm" (default), "elastic_net", or "scm".
        search_mode (str): Search strategy for treatment selection.
            - "exhaustive": test all C(n,k) combinations (default for small problems).
            - "ranking": screen geos individually, pick top-k, re-evaluate.
            - "auto": use ranking if C(n,k) > 1000, exhaustive otherwise.
        alpha (float or list): ElasticNet regularization strength. 
            Single value (default: 0.01) or list for grid search (e.g. [0.001, 0.01, 0.1]).
        l1_ratio (float or list): ElasticNet L1/L2 mixing ratio.
            Single value (default: 0.5) or list for grid search (e.g. [0.2, 0.5, 0.8]).
        verbose (bool): Whether to print running logs.
        df (pd.DataFrame, optional): Pre-loaded DataFrame. When provided, skips CSV I/O.

    Returns:
        list: List of dictionaries containing the best cluster splits.
    """
    if df is not None:
        df = df.copy()
    else:
        if filepath is None:
            raise ValueError("Either 'filepath' or 'df' must be provided.")
        df = pd.read_csv(filepath)
        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
        df = df.dropna(subset=[date_col])

    if start_date is not None:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.to_datetime(end_date)]
    
    import warnings
    with warnings.catch_warnings():
        try:
            from pandas.errors import PerformanceWarning
            warnings.simplefilter("ignore", category=PerformanceWarning)
        except ImportError:
            pass
        df = df.groupby(date_col).sum(numeric_only=True).reset_index()
        
    df = df.copy()  # Consolidate memory layout to defragment the DataFrame
    df = df.sort_values(date_col).reset_index(drop=True)

    if geos is None:
        geos = [col for col in df.columns if col != date_col]

    # Build grids from parameters (accept single value or list)
    alpha_grid = alpha if isinstance(alpha, list) else [alpha]
    l1_grid = l1_ratio if isinstance(l1_ratio, list) else [l1_ratio]
    if method == "scm":
        alpha_grid = [0.0]
        l1_grid = [0.0]

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
            method=method,
            alpha_grid=alpha_grid, l1_grid=l1_grid,
            verbose=verbose, desc="Screening geos", n_jobs=n_jobs,
            check_oof=check_oof,
        )

        # Sort and pick top-k
        phase1_results.sort(key=lambda x: x["placebo_score"])
        top_k = [r["treatment"][0] for r in phase1_results[:n_treatment]]

        if verbose:
            print(f"Top {n_treatment} candidates: {top_k}")
            print("Re-evaluating with proper control exclusions...\n")

        # Phase 2: Re-evaluate with cross-exclusion
        phase2_combos = [[t] for t in top_k]
        results = _evaluate_combinations(
            df, geos, phase2_combos,
            all_treatment_geos=set(top_k),
            method=method,
            alpha_grid=alpha_grid, l1_grid=l1_grid,
            verbose=False, desc=None, n_jobs=n_jobs,
            check_oof=check_oof,
        )

        clusters = _build_clusters(results, method)
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
        method=method,
        alpha_grid=alpha_grid, l1_grid=l1_grid,
        verbose=verbose,
        desc="Evaluating Combinations" if fixed_treatment is None else "Evaluating Fixed",
        n_jobs=n_jobs,
        check_oof=check_oof,
    )

    results = sorted(results, key=lambda x: x["placebo_score"])
    clusters = _build_clusters(results, method)

    if verbose and show_results:
        _print_results(clusters, is_fixed=(fixed_treatment is not None))

    return clusters if fixed_treatment else clusters[:10]


# ── Internal helpers ──────────────────────────────────────────────────────

def _fit_scm_weights(df_seg, treatment_comb, control_pool, method, a, l1):
    """
    Fit donor weights on df_seg for the given method/alpha/l1.
    Returns (final_controls, final_weights). Raises on failure.
    """
    use_penalized_scm = (method == "penalized_scm")
    use_en = (method == "elastic_net")

    if use_penalized_scm:
        X_syn = df_seg[control_pool].values.astype(float)
        y_syn = df_seg[list(treatment_comb)].mean(axis=1).values.astype(float)
        y_mean = y_syn.mean() or 1e-10
        X_mean = np.where(X_syn.mean(axis=0) == 0, 1e-10, X_syn.mean(axis=0))
        w_var = cp.Variable(len(control_pool))
        penalty = a * l1 * cp.norm1(w_var) + a * (1 - l1) * cp.sum_squares(w_var)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(y_syn / y_mean - (X_syn / X_mean) @ w_var) + penalty),
            [w_var >= 0],
        )
        prob.solve(solver=cp.SCS, verbose=False)
        w_vals = np.array(w_var.value).flatten()
        valid = [(c, wv) for c, wv in zip(control_pool, w_vals) if wv > 0.001]
        if valid:
            total_w = sum(wv for _, wv in valid)
            return [c for c, _ in valid], [wv / total_w for _, wv in valid]
        return control_pool[:], [1.0 / len(control_pool)] * len(control_pool)

    else:
        df_t = log_diff_transform(df_seg, list(treatment_comb) + control_pool)
        y = df_t[list(treatment_comb)].mean(axis=1).values
        X = df_t[control_pool].values
        X_scaled, _ = scale_data(X)

        if use_en:
            model = ElasticNet(alpha=a, l1_ratio=l1, max_iter=10000, tol=1e-2)
            model.fit(X_scaled, y)
            coefs = model.coef_
            selected = [control_pool[i] for i in range(len(control_pool)) if coefs[i] > 0]
            if not selected:
                selected = [control_pool[int(np.argmax(np.abs(coefs)))]]
        else:
            selected = control_pool[:]

        X_syn = df_seg[selected].values.astype(float)
        y_syn = df_seg[list(treatment_comb)].mean(axis=1).values.astype(float)
        y_mean = y_syn.mean() or 1e-10
        X_mean = np.where(X_syn.mean(axis=0) == 0, 1e-10, X_syn.mean(axis=0))
        w_syn = cp.Variable(len(selected))
        prob_syn = cp.Problem(
            cp.Minimize(cp.sum_squares(y_syn / y_mean - (X_syn / X_mean) @ w_syn)),
            [w_syn >= 0, cp.sum(w_syn) == 1],
        )
        prob_syn.solve(solver=cp.SCS, verbose=False)
        w_vals = np.array(w_syn.value).flatten()
        valid = [(c, wv) for c, wv in zip(selected, w_vals) if wv > 0.001]
        if valid:
            total_w = sum(wv for _, wv in valid)
            return [c for c, _ in valid], [wv / total_w for _, wv in valid]
        return selected[:], [1.0 / len(selected)] * len(selected)


def _eval_single_combination(args):
    """
    Evaluate a single treatment combination.

    When check_oof=True (default): splits the pre-period 70/30, fits the SCM on
    the train portion, applies the resulting weights to the validation portion
    (where true lift = 0), and runs CMBB on the residuals.  The score is
    max(|CI_lower%|, |CI_upper%|) — penalises both wide CIs and spurious lifts.

    When check_oof=False: ranks by SER (std_residual / correlation), the
    original in-sample metric.

    In both cases the inference weights come from a full pre-period fit.
    """
    treatment_comb, df_dict, geos, clean_all_treatments, method, alpha_grid, l1_grid, check_oof = args

    df = pd.DataFrame(df_dict)
    treatment_comb = [str(x).strip() for x in list(treatment_comb)]
    forbidden_geos = set(treatment_comb).union(clean_all_treatments)
    control_pool = [str(g).strip() for g in geos if str(g).strip() not in forbidden_geos]

    if not control_pool:
        return None

    n_rows = len(df)
    n_train = int(n_rows * 0.7)
    n_val = n_rows - n_train
    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:]

    try:
        best_local = None

        for a in alpha_grid:
            for l1 in l1_grid:
                try:
                    # Full pre-period fit → weights used for inference
                    final_controls, final_weights = _fit_scm_weights(
                        df, treatment_comb, control_pool, method, a, l1
                    )

                    # In-sample stats on full period (for reporting)
                    df_t = log_diff_transform(df, treatment_comb + final_controls)
                    y_full = df_t[treatment_comb].mean(axis=1).values
                    y_pred_full = df_t[final_controls].values @ np.array(final_weights)
                    residual = y_full - y_pred_full
                    rmspe = float(np.sqrt(np.mean(residual ** 2)))
                    std_res = float(np.std(residual))
                    corr = (
                        float(np.corrcoef(y_full, y_pred_full)[0, 1])
                        if np.std(y_full) > 0 and np.std(y_pred_full) > 0
                        else 0.0
                    )
                    ser = std_res / (corr + 1e-6)

                    # Placebo backtest: train-portion fit → val-portion CMBB
                    placebo_score = ser  # default fallback
                    if check_oof and n_val >= 6:
                        try:
                            score_controls, score_weights = _fit_scm_weights(
                                df_train, treatment_comb, control_pool, method, a, l1
                            )
                            y_val = df_val[treatment_comb].mean(axis=1).values
                            synthetic_val = df_val[score_controls].values @ np.array(score_weights)
                            effect_val = y_val - synthetic_val
                            post_synth_safe = np.where(synthetic_val <= 0, 1e-10, synthetic_val)
                            bs = bootstrap_significance(
                                effect_val, post_synth_safe, n_boot=300, random_state=42
                            )
                            placebo_score = max(abs(bs["ci_lower_pct"]), abs(bs["ci_upper_pct"]))
                        except Exception:
                            pass  # keep placebo_score = ser

                    candidate = {
                        "treatment": list(treatment_comb),
                        "control": final_controls,
                        "control_weights": final_weights,
                        "placebo_score": placebo_score,
                        "std_residual": std_res,
                        "rmspe": rmspe,
                        "correlation": corr,
                        "ser": ser,
                        "n_controls": len(final_controls),
                        "alpha": a,
                        "l1_ratio": l1,
                    }

                    if best_local is None or candidate["placebo_score"] < best_local["placebo_score"]:
                        best_local = candidate

                except Exception:
                    continue

        return best_local

    except Exception:
        return None


def _evaluate_combinations(df, geos, treatment_combinations, all_treatment_geos,
                           method, alpha_grid, l1_grid, verbose, desc, n_jobs=1, check_oof=True):
    """Run the evaluation loop over treatment combinations (sequential or parallel)."""
    
    # Ensure all_treatment_geos is a flat set of strings
    clean_all_treatments = set()
    if all_treatment_geos:
        for t in all_treatment_geos:
            if isinstance(t, list): clean_all_treatments.update([str(x).strip() for x in t])
            else: clean_all_treatments.add(str(t).strip())

    # Serialize DataFrame to dict for multiprocessing
    df_dict = df.to_dict(orient='list')
    
    # Build args for each combination
    args_list = [
        (tc, df_dict, geos, clean_all_treatments, method, alpha_grid, l1_grid, check_oof)
        for tc in treatment_combinations
    ]

    n_total = len(treatment_combinations)

    # Resolve n_jobs: auto-detect optimal workers based on workload
    import os
    cpu_count = os.cpu_count() or 1
    max_safe_jobs = max(cpu_count // 2, 1)
    if n_jobs is None:
        # Auto: parallelize only if workload justifies the overhead
        n_jobs = max_safe_jobs if n_total > 50 else 1
    elif n_jobs == -1:
        n_jobs = max_safe_jobs
    elif n_jobs == 0:
        n_jobs = 1
    elif n_jobs > max_safe_jobs:
        n_jobs = max_safe_jobs
    
    if n_jobs == 1:
        # Sequential mode (original behavior)
        if verbose and desc:
            try:
                from tqdm import tqdm
                iterator = tqdm(args_list, desc=desc, leave=True)
            except ImportError:
                iterator = args_list
                print(f"{desc}: {n_total} combinations...")
        else:
            iterator = args_list

        results = []
        for args in iterator:
            result = _eval_single_combination(args)
            if result is not None:
                results.append(result)
    else:
        # Parallel mode
        from concurrent.futures import ProcessPoolExecutor
        
        effective_jobs = min(n_jobs, n_total)
        if verbose and desc:
            unit = "geos" if all(len(tc) == 1 for tc in treatment_combinations) else "combinations"
            print(f"{desc}: {n_total:,} {unit} | {effective_jobs} workers")
        
        results = []
        chunksize = max(1, n_total // (effective_jobs * 4))
        try:
            from tqdm import tqdm as _tqdm
            _wrap = (lambda it, **kw: _tqdm(it, **kw)) if verbose else (lambda it, **kw: it)
        except ImportError:
            _wrap = lambda it, **kw: it
        with ProcessPoolExecutor(max_workers=effective_jobs) as executor:
            for result in _wrap(
                executor.map(_eval_single_combination, args_list, chunksize=chunksize),
                total=n_total, desc=desc, leave=True,
            ):
                if result is not None:
                    results.append(result)

    if len(results) == 0:
        raise ValueError("No valid combinations found.")

    return results


def _build_clusters(results, method):
    """Sort results and build cluster list."""
    results = sorted(results, key=lambda x: x["placebo_score"])

    clusters = []
    for res in results:
        clusters.append({
            "treatment": res["treatment"],
            "control": res["control"],
            "control_weights": res["control_weights"],
            "correlation": res["correlation"],
            "std_residual": res["std_residual"],
            "rmspe": res["rmspe"],
            "ser": res["ser"],
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
        print("     It minimizes the 'Synchronization Error Ratio (SER)' (Std Residual / Correlation), balancing low error with high synchronization.")
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
        print(f"\n{label_prefix} {i} | Treatment: [{treatment_str}] | Correlation: {c['correlation']:.4f} | RMSPE: {c['rmspe']:.4f} | Std Residual: {c['std_residual']:.4f} | SER: {c['ser']:.4f}")
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