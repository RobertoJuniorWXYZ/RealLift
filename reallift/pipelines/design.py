import pandas as pd
import numpy as np
import warnings
from ..geo.discovery import discover_geo_clusters
from ..geo.duration import estimate_duration
from ..config.defaults import DEFAULT_TREATMENT_PCTS, DEFAULT_EXPERIMENT_DAYS
from .shared import _run_oof_refinement_single
from .reporting import _print_scenario_table, _build_comparison, _print_comparison_table
from ..utils.reporting import generate_doe_report

def design_of_experiments(
    filepath,
    date_col,
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
    n_jobs=None,
    verbose=True,
    save_pdf=False,
    pdf_name='doe_report.pdf',
    logo=None
) -> dict:
    """
    Design of Experiments (DoE) — Scenario analysis for GeoLift experiments.

    Automatically generates experiment scenarios at different treatment allocation
    percentages (default: 10%, 20%, 30%), running cluster discovery and duration
    estimation for each. Displays a comparative MDE table to help decide the
    optimal trade-off between sensitivity and intervention cost.
    """
    valid_types = ["synthetic_control", "matched_did"]
    if experiment_type not in valid_types:
        raise ValueError(f"Invalid experiment_type '{experiment_type}'. Allowed types are 'synthetic_control' and 'matched_did'.")

    # 1. Detect available geos
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
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
        print(f"Search mode: {search_mode}")
        print(f"Pre-treatment period: {pre_start} → {end_date}")
        print(f"Experiment duration: {experiment_days}\n")

    # 3. Global screening (only for ranking mode)
    max_n_treat = max(c["n_treatment"] for c in scenario_configs)
    has_fixed = any(c["fixed"] for c in scenario_configs)

    global_ranking = None
    if not has_fixed and search_mode == "ranking":
        from reallift.geo.discovery import _evaluate_combinations

        if verbose:
            print("Screening all geos individually...")

        df_screen = pd.read_csv(filepath)
        df_screen[date_col] = pd.to_datetime(df_screen[date_col], format='mixed', dayfirst=True, errors='coerce')
        df_screen = df_screen.dropna(subset=[date_col])
        if start_date:
            df_screen = df_screen[df_screen[date_col] >= pd.to_datetime(start_date)]
        if end_date:
            df_screen = df_screen[df_screen[date_col] <= pd.to_datetime(end_date)]
        df_screen = df_screen.groupby(date_col).sum(numeric_only=True).reset_index()
        df_screen = df_screen.sort_values(date_col).reset_index(drop=True)

        alpha_grid = [0.01]
        l1_grid = [0.5]

        phase1_combos = [[g] for g in geos]
        phase1_results = _evaluate_combinations(
            df_screen, geos, phase1_combos,
            all_treatment_geos=set(),
            use_elasticnet=use_elasticnet,
            alpha_grid=alpha_grid, l1_grid=l1_grid,
            verbose=verbose, desc="Screening geos", n_jobs=n_jobs
        )
        phase1_results.sort(key=lambda x: x["ser"])
        global_ranking = [r["treatment"][0] for r in phase1_results]

        if verbose:
            print(f"Global ranking (top {max_n_treat}): {global_ranking[:max_n_treat]}\n")

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
                    n_jobs=n_jobs
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
                            verbose=False, show_results=False
                        )
                        current_cluster = candidate_eval[0].copy()
                    except Exception:
                        continue
                    
                    best_cluster, best_cv_row, passed, iters = _run_oof_refinement_single(
                        current_cluster, filepath, date_col, df_pre, start_date, end_date, n_folds, experiment_type=experiment_type
                    )
                    
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
                                verbose=False, show_results=False
                            )
                            current_cluster = candidate_eval[0].copy()
                        except Exception:
                            iter_results.append({"candidate": candidate, "passed": False})
                            continue
                        
                        best_cluster, best_cv_row, passed, iters = _run_oof_refinement_single(
                            current_cluster, filepath, date_col, df_pre, start_date, end_date, n_folds, experiment_type=experiment_type
                        )
                        
                        r2_t = best_cv_row["r2_test"]
                        gap_t = best_cv_row["r2_train"] - r2_t
                        
                        iter_results.append({
                            "candidate": candidate, "passed": passed,
                            "raw": current_cluster, "best": best_cluster, "cv": best_cv_row,
                            "r2": r2_t, "gap": gap_t
                        })
                    
                    # Consolidate (lock) newly passed clusters
                    newly_passed = [r for r in iter_results if r.get("passed", False)]
                    newly_failed = [r for r in iter_results if not r.get("passed", False)]
                    
                    for r in newly_passed:
                        locked_clusters.append(r)
                        locked_treatments.add(r["candidate"])
                        if r.get("best"):
                            locked_donors.update(r["best"].get("control", []))
                    
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
                                })
                            single_candidate_skips.extend(iter_skipped)
                        else:
                            # Flush any pending single-candidate buffer first
                            _flush_single_buffer(single_candidate_buffer, single_candidate_skips, len(locked_treatments) - sum(1 for r in newly_passed))
                            single_candidate_buffer.clear()
                            single_candidate_skips.clear()
                            
                            # Print multi-candidate iteration
                            cons_str = f" | {len(locked_treatments) - len(newly_passed)} consolidated" if locked_treatments - set(r["candidate"] for r in newly_passed) else ""
                            print(f"    Iter {iteration} | Testing {len(candidates_to_test)} candidates{cons_str}")
                            for r in iter_results:
                                status = "Approved" if r.get("passed") else "Failed  "
                                print(f"      [{status}] {r['candidate']:<10} R²={r.get('r2', 0):.4f}  Gap={r.get('gap', 0):.4f}")
                            
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
                    verbose=False, n_jobs=n_jobs
                )
                
                raw_items = []
                group_passed = True
                r2_sum = 0
                
                for i, cluster in enumerate(current_clusters):
                    treat_list = cluster['treatment']
                    treat_str = treat_list[0] if treat_list else "Unknown"
                    eval_cluster = cluster.copy()
                    
                    best_cluster, best_cv_row, passed, iters = _run_oof_refinement_single(
                        eval_cluster, filepath, date_col, df_pre, start_date, end_date, n_folds, experiment_type=experiment_type
                    )
                    
                    r2_t = best_cv_row["r2_test"]
                    gap_t = best_cv_row["r2_train"] - r2_t
                    r2_sum += r2_t
                    
                    if experiment_type == "matched_did":
                        if passed:
                            if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Optimal (R² = {r2_t:.4f}, Iters: {iters})")
                        else:
                            if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Failed strict rules. Best R² = {r2_t:.4f}, Iters: {iters}")
                            group_passed = False
                    else:
                        if passed:
                            if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Optimal (OOF R² = {r2_t:.4f}, Gap = {gap_t:.4f}, Iters: {iters})")
                        else:
                            if verbose: print(f"      - Cluster {i} ({treat_str:<10}) Failed strict rules. (OOF R² = {r2_t:.4f}, Gap = {gap_t:.4f}, Iters: {iters})")
                            group_passed = False
                    
                    raw_items.append({
                        "raw": eval_cluster, "best": best_cluster,
                        "cv": best_cv_row, "passed": passed
                    })
                
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
            verbose=False
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
