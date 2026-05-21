import pandas as pd
import numpy as np
import warnings
from ._discovery import discover_geo_clusters
from ._duration import estimate_duration
from ..config.defaults import DEFAULT_TREATMENT_PCTS, DEFAULT_EXPERIMENT_DAYS
from ._shared import _run_oof_refinement_single
from ._reporting import _print_scenario_table, _build_comparison, _print_comparison_table
from ._reporting import generate_doe_report
from ._bootstrap import bootstrap_significance
import cvxpy as cp

def _check_ghost_lift_oos(clusters, df_pre, date_col, experiment_days):
    """
    Out-of-Sample Backtest for Ghost Lift detection via Weekly-Aligned Bootstrap.

    Splits the pre-treatment period into Train ([:-d]) and Test ([-d:]).
    Trains a strictly convex, level-normalized Synthetic Control on Train,
    predicts on Test, then segments the effect series into complete calendar
    weeks (Monday→Sunday).

    A single i.i.d. bootstrap is run over the K weekly mean effects.
    Weekly aggregation naturally removes within-week autocorrelation,
    making i.i.d. resampling valid.  If the 95 % CI of the aggregate lift
    does NOT cross zero, a Ghost Lift is detected.

    Supports single cluster or list of clusters (consolidated check).

    Returns True if a Ghost Lift is detected for any requested horizon.
    """
    _N_BOOT = 2000

    if not isinstance(clusters, list):
        clusters = [clusters]

    if not isinstance(experiment_days, (list, tuple)):
        eval_sizes = [int(experiment_days)]
    else:
        eval_sizes = sorted(int(d) for d in experiment_days)

    for d in eval_sizes:
        if d >= len(df_pre) - 5:  # Need enough training data
            continue

        train_df = df_pre.iloc[:-d]
        test_df = df_pre.iloc[-d:]

        y_test_total = np.zeros(d)
        y_pred_total = np.zeros(d)

        for cluster in clusters:
            treat_cols = cluster["treatment"]
            controls = cluster["control"]

            if len(controls) == 0:
                continue

            y_train = train_df[treat_cols].mean(axis=1).values.astype(float)
            X_train = train_df[controls].values.astype(float)

            # Train
            y_mean_train = y_train.mean() if y_train.mean() != 0 else 1e-10
            X_mean_train = X_train.mean(axis=0)
            X_mean_train[X_mean_train == 0] = 1e-10

            y_norm_train = y_train / y_mean_train
            X_norm_train = X_train / X_mean_train

            w_syn = cp.Variable(X_norm_train.shape[1])
            obj_syn = cp.Minimize(cp.sum_squares(y_norm_train - (X_norm_train @ w_syn)))
            cons_syn = [w_syn >= 0, cp.sum(w_syn) == 1]
            prob_syn = cp.Problem(obj_syn, cons_syn)
            try:
                prob_syn.solve(solver=cp.SCS, verbose=False)
                w_vals = np.array(w_syn.value).flatten()
                w_vals[w_vals < 0] = 0.0
                sum_w = np.sum(w_vals)
                if sum_w > 0:
                    w_vals = w_vals / sum_w
            except Exception:
                w_vals = np.ones(X_norm_train.shape[1]) / X_norm_train.shape[1]

            # Test
            y_test = test_df[treat_cols].mean(axis=1).values.astype(float)
            X_test = test_df[controls].values.astype(float)

            X_norm_test = X_test / X_mean_train
            y_pred = (X_norm_test @ w_vals) * y_mean_train

            y_test_total += y_test
            y_pred_total += y_pred

        # ── Weekly-Aligned Bootstrap ──────────────────────────────────────
        if np.sum(y_pred_total) <= 0:
            continue

        effect = y_test_total - y_pred_total
        test_dates = pd.to_datetime(test_df[date_col].values)

        # Group by ISO calendar week (Monday-anchored)
        iso_cal = test_dates.isocalendar()
        week_ids = [f"{y}-W{w:02d}" for y, w in zip(iso_cal.year, iso_cal.week)]

        # Aggregate effect and baseline per week
        week_effects = {}
        week_baselines = {}
        for i, wid in enumerate(week_ids):
            week_effects.setdefault(wid, []).append(effect[i])
            week_baselines.setdefault(wid, []).append(y_pred_total[i])

        # Keep only complete weeks (exactly 7 days)
        weekly_mean_effects = []
        weekly_mean_baselines = []
        for wid in week_effects:
            if len(week_effects[wid]) == 7:
                weekly_mean_effects.append(np.mean(week_effects[wid]))
                weekly_mean_baselines.append(np.mean(week_baselines[wid]))

        if len(weekly_mean_effects) < 2:
            # Not enough complete weeks — fall back to MBB on full series
            boot = bootstrap_significance(effect, y_pred_total, random_state=42)
            if boot["ci_lower_total_pct"] > 0 or boot["ci_upper_total_pct"] < 0:
                return True
            continue

        weekly_mean_effects = np.array(weekly_mean_effects)
        weekly_mean_baselines = np.array(weekly_mean_baselines)
        n_weeks = len(weekly_mean_effects)

        # i.i.d. bootstrap over weekly means (autocorrelation already
        # removed by the weekly aggregation, so blocks are unnecessary)
        rng = np.random.default_rng(42)
        boot_pcts = np.empty(_N_BOOT)

        for b in range(_N_BOOT):
            idx = rng.choice(n_weeks, size=n_weeks, replace=True)
            sum_eff = weekly_mean_effects[idx].sum()
            sum_base = weekly_mean_baselines[idx].sum()
            boot_pcts[b] = sum_eff / sum_base if abs(sum_base) > 1e-10 else 0.0

        ci_lower = np.percentile(boot_pcts, 2.5)
        ci_upper = np.percentile(boot_pcts, 97.5)

        if ci_lower > 0 or ci_upper < 0:
            return True  # Ghost Lift detected

    return False  # Passed all horizons

def design_of_experiments(
    filepath=None,
    date_col=None,
    start_date=None,
    end_date=None,
    geos=None,
    pct_treatment=None,
    fixed_treatment=None,
    mde=None,
    experiment_days=DEFAULT_EXPERIMENT_DAYS,
    n_folds=5,
    search_mode="ranking",
    experiment_type="synthetic_control",
    use_elasticnet=False,
    check_ghost_lift=True,
    n_jobs=None,
    verbose=True,
    save_pdf=False,
    pdf_name='doe_report.pdf',
    logo=None,
    df=None
) -> dict:
    """
    Design of Experiments (DoE) — Scenario analysis for GeoLift experiments.

    Automatically generates experiment scenarios at different treatment allocation
    percentages (default: 10%, 20%, 30%), running cluster discovery and duration
    estimation for each. Displays a comparative MDE table to help decide the
    optimal trade-off between sensitivity and intervention cost.
    
    Includes robust multivariate evaluation: Time-Series Cross-Validation (OOF) 
    combined with a strict Consolidated Out-of-Sample (OOS) Ghost Lift detection 
    to ensure the absence of structural aggregation bias.
    
    Parameters:
        ...
        check_ghost_lift (bool): If True, strictly enforces the OOS Ghost Lift check 
            on both individual candidates and the consolidated group. Any candidate 
            that induces an additive bias across the synthetic portfolio is rejected.
            Default is True.
        df (pd.DataFrame, optional): Pre-loaded DataFrame. When provided, skips CSV I/O.
    """
    valid_types = ["synthetic_control", "matched_did"]
    if experiment_type not in valid_types:
        raise ValueError(f"Invalid experiment_type '{experiment_type}'. Allowed types are 'synthetic_control' and 'matched_did'.")

    # 1. Detect available geos
    if df is not None:
        df = df.copy()
    else:
        if filepath is None:
            raise ValueError("Either 'filepath' or 'df' must be provided.")
        df = pd.read_csv(filepath)
        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
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
        print(f"Pre-treatment period: {pre_start} → {end_date}")
        print(f"Experiment duration: {experiment_days}\n")
        
        # Info about OOS Backtest
        if check_ghost_lift:
            print(f"  [Info] OOS Backtest will validate Ghost Lifts across ALL requested horizons at the end of the historical series.\n")
        else:
            print(f"  [Warning] OOS Ghost Lift validation is DISABLED.\n")

    # 3. Global Base Ranking
    max_n_treat = max(c["n_treatment"] for c in scenario_configs)
    has_fixed = any(c["fixed"] for c in scenario_configs)

    global_ranking = None
    if not has_fixed and search_mode == "ranking":
        from ._discovery import _evaluate_combinations
        
        df_rank = df.copy()
        if start_date:
            df_rank = df_rank[df_rank[date_col] >= pd.to_datetime(start_date)]
        if end_date:
            df_rank = df_rank[df_rank[date_col] <= pd.to_datetime(end_date)]
            
        with warnings.catch_warnings():
            try:
                from pandas.errors import PerformanceWarning
                warnings.simplefilter("ignore", category=PerformanceWarning)
            except ImportError:
                pass
            df_rank = df_rank.groupby(date_col).sum(numeric_only=True).reset_index()
            
        df_rank = df_rank.copy()  # Consolidate memory layout to defragment the DataFrame
        df_rank = df_rank.sort_values(date_col).reset_index(drop=True)
        
        if verbose: print("\n  Evaluating full historical data for global ranking...")
        valid_combos = [[g] for g in geos]
        phase1_results_full = _evaluate_combinations(
            df_rank, geos, valid_combos,
            all_treatment_geos=set(),
            use_elasticnet=use_elasticnet,
            alpha_grid=[0.01], l1_grid=[0.5],
            verbose=False, desc="Ranking Geos", n_jobs=n_jobs
        )
        phase1_results_full.sort(key=lambda x: x["ser"])
        global_ranking = [r["treatment"][0] for r in phase1_results_full]

        if verbose:
            print(f"  Global ranking (top {max_n_treat}): {global_ranking[:max_n_treat]}\n")

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
                best_groups = [{"treatment": config["fixed"]}]
                search_mode_used = "fixed"
            elif global_ranking is not None:
                # Greedy Sequential Search mode handles discovery inside OOF loop
                best_groups = None
                search_mode_used = "ranking"
            else:
                # Exhaustive mode
                best_groups = discover_geo_clusters(
                    filepath=filepath,
                    date_col=date_col,
                    geos=geos,
                    n_treatment=n_treat,
                    start_date=start_date,
                    end_date=end_date,
                    use_elasticnet=use_elasticnet,
                    search_mode=search_mode,
                    verbose=True,
                    show_results=False,
                    n_jobs=n_jobs,
                    df=df
                )
                search_mode_used = "exhaustive"
                if verbose:
                    print(f"  Identified top {len(best_groups)} combinations. Proceeding to OOF Refinement.")
        except Exception as e:
            if verbose:
                print(f"  [Error] Failed to find clusters: {e}\n")
            scenarios.append({
                "pct_treatment": pct,
                "n_treatment": n_treat,
                "clusters": None,
                "duration": None,
                "error": str(e),
            })
            continue

        # 4b. OOF Refinement & Cross-validation
        refined_clusters = []
        cv_rows = []
        df_pre = df.copy()
        
        run_as_matched_did = (experiment_type == "matched_did")

        if global_ranking is not None:
            if run_as_matched_did:
                if verbose: print("\n  [Matched DiD] Evaluating all candidates to maximize correlation...")
                
                all_evals = []
                for candidate in geos:
                    try:
                        candidate_eval = discover_geo_clusters(
                            filepath=filepath, date_col=date_col, geos=geos,
                            fixed_treatment=[candidate], start_date=start_date, end_date=end_date,
                            use_elasticnet=use_elasticnet,
                            verbose=False, show_results=False, df=df
                        )
                        current_cluster = candidate_eval[0].copy()
                    except Exception:
                        continue
                    
                    best_cluster, best_cv_row, passed, iters = _run_oof_refinement_single(
                        current_cluster, filepath, date_col, df_pre, start_date, end_date, n_folds, experiment_days=experiment_days, experiment_type=experiment_type, df=df
                    )
                    
                    ghost_lift = False
                    if passed:
                        if experiment_type != "matched_did":
                            ghost_lift = _check_ghost_lift_oos(best_cluster, df_pre, date_col, experiment_days)
                        if ghost_lift:
                            passed = False
                            
                    if not passed:
                        continue
                    
                    all_evals.append({
                        "raw": current_cluster, "best": best_cluster, "cv": best_cv_row,
                        "candidate": candidate, "r2": best_cv_row["r2_test"], "iters": iters
                    })
                
                # Sort descending by maximum reached holistic R²
                all_evals.sort(key=lambda x: x["r2"], reverse=True)
                
                # Identify the intended treatment set for this scenario based on Matched DiD ranking
                intended_treatment_pool = [item["candidate"] for item in all_evals[:n_treat]]
                
                locked_treatments = []
                
                # Selection Loop: pick candidates and ensure their donor pools are clean 
                # from OTHER treatments in this specific scenario.
                for item in all_evals:
                    if len(refined_clusters) >= n_treat: break
                    candidate = item["candidate"]
                    
                    if candidate in intended_treatment_pool:
                        # PURIFY: Remove any other unit that belongs to the treatment pool from this candidate's donor set
                        best_mod = item["best"].copy()
                        others = [t for t in intended_treatment_pool if t != candidate]
                        
                        clean_controls = [d for d in best_mod["control"] if d not in others]
                        
                        # Update weights to be uniform over the clean pool (DiD Standard)
                        if len(clean_controls) > 0:
                            best_mod["control"] = clean_controls
                            best_mod["control_weights"] = [1.0/len(clean_controls)] * len(clean_controls)
                            
                            item["best"] = best_mod
                            if verbose: print(f"    - Candidate: {candidate:<10} Selected (Cleaned, R² = {item['r2']:.4f})      ")
                            refined_clusters.append(item)
                            locked_treatments.append(candidate)
                        
                # Sort refined_clusters so final output is strict R2 order
                refined_clusters.sort(key=lambda x: x["r2"], reverse=True)
                            
            else:
                if verbose: print("\n  [Greedy Lock Search]\n")
                
                # Initialize: top n_treat candidates from the SER ranking
                current_candidates = list(global_ranking[:n_treat])
                next_rank_idx = n_treat
                
                max_iterations = len(global_ranking)
                locked_clusters = []       # Consolidated clusters (frozen)
                locked_treatments = set()  # Treatment geos already consolidated
                locked_donors = set()      # Donors used by consolidated clusters
                all_failed = []            # All failed candidates across iterations (for fallback)
                
                found = False
                iteration = 0
                # Track iterations that test a single candidate (for compact display)
                single_candidate_buffer = []
                single_candidate_skips = []
                
                def _flush_single_buffer(buf, skips, n_consolidated):
                    """Print compacted single-candidate iteration results."""
                    if not buf:
                        return
                    iter_range = f"Iter {buf[0]['iter']}" if len(buf) == 1 else f"Iter {buf[0]['iter']}–{buf[-1]['iter']}"
                    print(f"    {iter_range} | Testing 1 candidate each | {n_consolidated} consolidated")
                    # Group failures into compact lines (4 per line)
                    failed_items = [f"{r['candidate']} ({r['r2']:.2f})" for r in buf if not r['passed']]
                    passed_items = [r for r in buf if r['passed']]
                    for r in passed_items:
                        print(f"      [Approved] {r['candidate']:<10} R²={r['r2']:.4f}  Gap={r['gap']:.4f}")
                    if failed_items:
                        for j in range(0, len(failed_items), 4):
                            chunk = " | ".join(failed_items[j:j+4])
                            print(f"      [Failed] {chunk}")
                    if skips:
                        skip_names = ", ".join(skips)
                        print(f"      → Skipped {len(skips)} donor-blocked ({skip_names})")
                
                for iteration in range(1, max_iterations + 1):
                    # Only test non-locked candidates
                    candidates_to_test = [c for c in current_candidates if c not in locked_treatments]
                    
                    if not candidates_to_test:
                        break
                    
                    is_single = len(candidates_to_test) == 1
                    
                    if verbose and not is_single:
                        _flush_single_buffer(single_candidate_buffer, single_candidate_skips, len(locked_treatments))
                        single_candidate_buffer.clear()
                        single_candidate_skips.clear()
                        cons_str = f" | {len(locked_treatments)} consolidated" if locked_treatments else ""
                        print(f"    Iter {iteration} | Testing {len(candidates_to_test)} candidates{cons_str}", flush=True)
                    
                    iter_results = []
                    
                    for candidate in candidates_to_test:
                        # Exclude all other treatments (locked + other new) from donor pool
                        other_treatments = set(current_candidates) - {candidate}
                        available_geos = [g for g in geos if g not in other_treatments]
                        
                        try:
                            candidate_eval = discover_geo_clusters(
                                filepath=filepath, date_col=date_col, geos=available_geos,
                                fixed_treatment=[candidate], start_date=start_date, end_date=end_date,
                                use_elasticnet=use_elasticnet,
                                verbose=False, show_results=False, df=df
                            )
                            current_cluster = candidate_eval[0].copy()
                        except Exception:
                            iter_results.append({"candidate": candidate, "passed": False})
                            continue
                        
                        best_cluster, best_cv_row, passed, iters = _run_oof_refinement_single(
                            current_cluster, filepath, date_col, df_pre, start_date, end_date, n_folds, experiment_days=experiment_days, experiment_type=experiment_type, df=df
                        )
                        
                        ghost_lift = False
                        if passed and experiment_type != "matched_did" and check_ghost_lift:
                            ghost_lift = _check_ghost_lift_oos(best_cluster, df_pre, date_col, experiment_days)
                            if ghost_lift:
                                passed = False
                        
                        r2_t = best_cv_row["r2_test"]
                        gap_t = best_cv_row["r2_train"] - r2_t
                        
                        iter_results.append({
                            "candidate": candidate, "passed": passed,
                            "raw": current_cluster, "best": best_cluster, "cv": best_cv_row,
                            "r2": r2_t, "gap": gap_t, "ghost_lift": ghost_lift
                        })
                        
                        if verbose and not is_single:
                            if passed:
                                status = "Approved"
                            else:
                                if ghost_lift:
                                    status = "GhostLift"
                                elif best_cv_row is not None and r2_t >= 0.6:
                                    status = "FailedR2  "
                                else:
                                    status = "Failed  "
                            print(f"      [{status}] {candidate:<10} R²={r2_t:.4f}  Gap={gap_t:.4f}", flush=True)
                    
                    # Consolidate (lock) newly passed clusters
                    newly_passed = [r for r in iter_results if r.get("passed", False)]
                    newly_failed = [r for r in iter_results if not r.get("passed", False)]
                    
                    # Sort by R2 to prioritize best candidates
                    newly_passed.sort(key=lambda x: x["r2"], reverse=True)
                    
                    for r in newly_passed:
                        # Trial group (currently locked + this candidate)
                        trial_clusters = [item["best"] for item in locked_clusters] + [r["best"]]
                        
                        consolidated_ghost_lift = False
                        if experiment_type != "matched_did" and check_ghost_lift and len(trial_clusters) > 1:
                            consolidated_ghost_lift = _check_ghost_lift_oos(trial_clusters, df_pre, date_col, experiment_days)
                        
                        if not consolidated_ghost_lift:
                            locked_clusters.append(r)
                            locked_treatments.add(r["candidate"])
                            if r.get("best"):
                                locked_donors.update(r["best"].get("control", []))
                        else:
                            # It caused a consolidated ghost lift, so we reject it
                            r["passed"] = False
                            r["ghost_lift"] = True
                            newly_failed.append(r)
                            if verbose and not is_single:
                                print(f"      [Rejected] {r['candidate']} caused a CONSOLIDATED Ghost Lift!")
                    
                    all_failed.extend([r for r in newly_failed if r.get("best") is not None])
                    
                    # Find replacements + track skipped
                    iter_skipped = []
                    new_candidates = sorted(locked_treatments)
                    
                    if len(locked_clusters) < n_treat:
                        while len(new_candidates) < n_treat and next_rank_idx < len(global_ranking):
                            next_geo = global_ranking[next_rank_idx]
                            next_rank_idx += 1
                            if next_geo in locked_treatments:
                                continue
                            if next_geo in locked_donors:
                                iter_skipped.append(next_geo)
                                continue
                            new_candidates.append(next_geo)
                    
                    # --- VERBOSE OUTPUT ---
                    if verbose:
                        if is_single:
                            # Buffer single-candidate iterations for compact display
                            for r in iter_results:
                                single_candidate_buffer.append({
                                    "iter": iteration,
                                    "candidate": r["candidate"],
                                    "passed": r.get("passed", False),
                                    "r2": r.get("r2", 0.0),
                                    "gap": r.get("gap", 0.0),
                                    "ghost_lift": r.get("ghost_lift", False)
                                })
                            single_candidate_skips.extend(iter_skipped)
                        else:
                            # Individual results were printed in real-time. Just print the footer.
                            n_cons = len(locked_clusters)
                            parts = [f"Consolidated {n_cons}"]
                            if iter_skipped:
                                skip_names = ", ".join(iter_skipped)
                                parts.append(f"Skipped {len(iter_skipped)} donor-blocked ({skip_names})")
                            print(f"      → {' | '.join(parts)}")
                    
                    # Check if all slots are filled
                    if len(locked_clusters) >= n_treat:
                        refined_clusters = [{"raw": r.get("raw"), "best": r["best"], "cv": r["cv"]} for r in locked_clusters[:n_treat]]
                        found = True
                        # Flush remaining buffer
                        if verbose:
                            _flush_single_buffer(single_candidate_buffer, single_candidate_skips, len(locked_treatments) - len(newly_passed))
                            single_candidate_buffer.clear()
                            single_candidate_skips.clear()
                            print(f"\n    [Success] All {n_treat} candidates consolidated by iteration {iteration}.")
                        break
                    
                    if len(new_candidates) < n_treat:
                        # Flush remaining buffer
                        if verbose:
                            _flush_single_buffer(single_candidate_buffer, single_candidate_skips, len(locked_treatments))
                            single_candidate_buffer.clear()
                            single_candidate_skips.clear()
                            print(f"\n    [Exhausted] Ranking exhausted after iteration {iteration}.")
                        break
                    
                    current_candidates = new_candidates
                
                if not found:
                    # Assemble: locked clusters + best failed candidates
                    refined_clusters = [{"raw": r.get("raw"), "best": r["best"], "cv": r["cv"]} for r in locked_clusters]
                    
                    remaining = n_treat - len(refined_clusters)
                    fallback_names = []
                    if remaining > 0 and all_failed:
                        all_failed.sort(key=lambda x: x["cv"]["r2_test"], reverse=True)
                        for r in all_failed[:remaining]:
                            refined_clusters.append({"raw": r.get("raw"), "best": r["best"], "cv": r["cv"]})
                            fallback_names.append(r["candidate"])
                    
                    if verbose:
                        fb_str = f" + {len(fallback_names)} fallback ({', '.join(fallback_names)})" if fallback_names else ""
                        print(f"\n    [Result] {len(locked_clusters)} consolidated{fb_str} to fill {n_treat} slots.")
                
                # ── DESIGN QUALITY ──
                if verbose:
                    r2_vals = []
                    for item in refined_clusters:
                        if item.get("cv") is not None:
                            r2_vals.append(max(item["cv"]["r2_test"], 0.01))
                    
                    if r2_vals:
                        quality_score = len(r2_vals) / sum(1.0 / v for v in r2_vals)
                        if quality_score >= 0.90:
                            rating = "Excellent"
                        elif quality_score >= 0.75:
                            rating = "Good"
                        elif quality_score >= 0.60:
                            rating = "Fair"
                        else:
                            rating = "Poor"
                        
                        n_consolidated = len(locked_clusters)
                        n_fallback = len(refined_clusters) - n_consolidated
                        
                        print(f"\n  DESIGN QUALITY")
                        print(f"  {'─' * 70}")
                        print(f"  Quality Score : {quality_score:.2f} [{rating}]{'':8}(harmonic mean of R² test)")
                        print(f"  Consolidated  : {n_consolidated}/{n_treat} ({n_consolidated/n_treat:.0%}){'':10}(passed strict OOF rules)")
                        if n_fallback > 0:
                            fb_details = []
                            for fn in fallback_names:
                                for item in refined_clusters:
                                    if item.get("best") and item["best"]["treatment"][0] == fn:
                                        fb_details.append(f"{fn}, R² test = {item['cv']['r2_test']:.4f}")
                            fb_str = "; ".join(fb_details) if fb_details else f"{n_fallback} cluster(s)"
                            print(f"  Fallback      : {n_fallback} cluster{'s' if n_fallback > 1 else ''}{'':10}({fb_str})")
                        print(f"  {'─' * 70}")
            
            clusters = [item["best"] for item in refined_clusters]
            cv_rows = [item["cv"] for item in refined_clusters]
            cv_summary = pd.DataFrame(cv_rows).reset_index(drop=True)
            
        else:
            if verbose: print(f"\n  [{'Matched DiD' if experiment_type == 'matched_did' else 'OOF Refinement'}]:")
            
            best_raw_items = None
            best_r2_sum = -float("inf")
            all_passed = False
            
            for group_idx, group in enumerate(best_groups):
                try_geos = group["treatment"]
                if search_mode_used == "exhaustive" and verbose:
                    print(f"\n    [Option {group_idx + 1}/{len(best_groups)}] Evaluating combination: {try_geos}")
                
                # Retrieve individual clusters for this combination
                current_clusters = discover_geo_clusters(
                    filepath=filepath, date_col=date_col, geos=geos,
                    fixed_treatment=try_geos, start_date=start_date, end_date=end_date,
                    use_elasticnet=use_elasticnet,
                    verbose=False, n_jobs=n_jobs, df=df
                )
                
                raw_items = []
                group_passed = True
                r2_sum = 0
                
                for i, cluster in enumerate(current_clusters):
                    treat_list = cluster['treatment']
                    treat_str = treat_list[0] if treat_list else "Unknown"
                    eval_cluster = cluster.copy()
                    
                    best_cluster, best_cv_row, passed, iters = _run_oof_refinement_single(
                        eval_cluster, filepath, date_col, df_pre, start_date, end_date, n_folds, experiment_days=experiment_days, experiment_type=experiment_type, df=df
                    )
                    
                    ghost_lift = False
                    if passed:
                        if experiment_type != "matched_did" and check_ghost_lift:
                            ghost_lift = _check_ghost_lift_oos(best_cluster, df_pre, date_col, experiment_days)
                        if ghost_lift:
                            passed = False
                    
                    r2_t = best_cv_row["r2_test"]
                    gap_t = best_cv_row["r2_train"] - r2_t
                    r2_sum += r2_t
                    
                    if experiment_type == "matched_did":
                        if passed:
                            if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Optimal (R² = {r2_t:.4f}, Iters: {iters})")
                        else:
                            if ghost_lift:
                                if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Failed: Ghost Lift Detected. Best R² = {r2_t:.4f}, Iters: {iters}")
                            else:
                                if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Failed strict rules. Best R² = {r2_t:.4f}, Iters: {iters}")
                            group_passed = False
                    else:
                        if passed:
                            if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Optimal (OOF R² = {r2_t:.4f}, Gap = {gap_t:.4f}, Iters: {iters})")
                        else:
                            if ghost_lift:
                                if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Failed: Ghost Lift Detected. (OOF R² = {r2_t:.4f}, Gap = {gap_t:.4f}, Iters: {iters})")
                            else:
                                if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Failed strict rules. (OOF R² = {r2_t:.4f}, Gap = {gap_t:.4f}, Iters: {iters})")
                            group_passed = False
                    
                    raw_items.append({
                        "raw": eval_cluster, "best": best_cluster,
                        "cv": best_cv_row, "passed": passed
                    })
                
                # Check for Consolidated Ghost Lift
                if group_passed and experiment_type != "matched_did" and check_ghost_lift and len(raw_items) > 1:
                    trial_clusters = [item["best"] for item in raw_items]
                    if _check_ghost_lift_oos(trial_clusters, df_pre, date_col, experiment_days):
                        group_passed = False
                        if verbose: print(f"    [Group Rejected] CONSOLIDATED Ghost Lift detected!")
                
                # Keep track of the best group in case all fail
                if best_raw_items is None or r2_sum > best_r2_sum:
                    best_raw_items = raw_items
                    best_r2_sum = r2_sum
                
                if group_passed:
                    all_passed = True
                    if search_mode_used == "exhaustive" and verbose:
                        print(f"    [Success] Found optimal passing combination: {try_geos}")
                    best_raw_items = raw_items
                    break
                else:
                    if search_mode_used == "exhaustive" and verbose:
                        print(f"    [Failed] Combination {try_geos} failed strict rules.")

            if experiment_type == "synthetic_control" and not all_passed:
                if verbose:
                    if search_mode_used == "exhaustive":
                        print("\n  [AUTO SENSOR] [Warning] All top combinations failed strict rules.")
                        print("                  Falling back to the combination with highest overall R².")
                    else:
                        print("\n  [AUTO SENSOR] [Warning] Some fixed clusters failed strict rules. Data may be too volatile for stable Synthetic Control.")
                        print("                  Consider re-evaluating the design with experiment_type='matched_did'.")
                    
            clusters = [item["best"] for item in best_raw_items]
            cv_rows = [item["cv"] for item in best_raw_items]
            cv_summary = pd.DataFrame(cv_rows).reset_index(drop=True)


        # 4c. Estimate duration (per-cluster + consolidated)
        duration = estimate_duration(
            filepath=filepath,
            date_col=date_col,
            clusters=clusters,
            mde=mde,
            experiment_days=experiment_days,
            start_date=start_date,
            end_date=end_date,
            verbose=False,
            df=df
        )

        # 4d. Compact verbose per scenario
        if verbose:
            _print_scenario_table(clusters, duration, mde, cv_summary, total_geos=n_geos, experiment_days=experiment_days, experiment_type=experiment_type)

        treatment_pool = []
        for c in clusters:
            treatment_pool.extend(c["treatment"])

        scenarios.append({
            "pct_treatment": pct,
            "n_treatment": n_treat,
            "treatment_pool": treatment_pool,
            "clusters": clusters,
            "duration": duration,
            "validation": cv_summary,
        })

    # 4. Final comparison table
    comparison = _build_comparison(scenarios, mde, experiment_days)

    if verbose:
        _print_comparison_table(comparison, mde, experiment_days=experiment_days)

    result = {
        "experiment_type": experiment_type,
        "scenarios": scenarios,
        "comparison": comparison,
    }

    if save_pdf:
        design_meta = {
            "n_geos": n_geos,
            "search_mode": search_mode,
            "experiment_type": experiment_type,
            "experiment_days": experiment_days,
            "mde": mde,
            "pre_start": pre_start,
            "end_date": end_date,
            "n_folds": n_folds,
        }
        generate_doe_report(
            pdf_name=pdf_name,
            doe_result=result,
            design_meta=design_meta,
            logo=logo
        )
        if verbose:
            print(f"\n  [PDF] Report saved to: {pdf_name}")

    return result
