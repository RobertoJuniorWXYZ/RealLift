import pandas as pd
import numpy as np
from scipy.stats import norm as _norm

def print_experiment_summary(results, date_col, experiment_type="synthetic_control", random_state=None):
    """
    Renders the final impact tables and consolidated results in the terminal.
    
    This function handles the heavy lifting of statistical aggregation, 
    confidence interval calculation, and MDE estimation for the experimental results.
    """
    if not results:
        return

    is_did = experiment_type == "matched_did"
    
    print("\n" + "=" * 70)
    print(" GEO EXPERIMENT RESULTS SUMMARY ".center(70, "="))
    print("=" * 70)
    
    df_first = results[0]["synthetic"]["df"]
    idx_treat = results[0]["synthetic"]["plotting_data"]["treatment_idx"]
    
    pre_start = df_first[date_col].iloc[0].strftime('%Y-%m-%d')
    pre_end = df_first[date_col].iloc[idx_treat-1].strftime('%Y-%m-%d')
    
    # Treatment window (as analyzed)
    post_start = idx_treat
    post_len = len(results[0]["synthetic"]["plotting_data"]["post_real"])
    post_days = post_len
    
    if post_len > 0:
        post_start_date = df_first[date_col].iloc[post_start].strftime('%Y-%m-%d')
        post_end_date = df_first[date_col].iloc[post_start + post_len - 1].strftime('%Y-%m-%d')
        print(f"\nPre-treatment period: {pre_start} → {pre_end}")
        print(f"Intervention period : {post_start_date} → {post_end_date} ({post_days} days)")
    else:
        print(f"\nPre-treatment period: {pre_start} → {pre_end}")
        print(f"Intervention period : N/A (Pre-only analysis)")
    
    n_clusters = len(results)
    num_treated = len(results[0]["cluster"]["treatment"])
    print(f"Clusters analyzed   : {n_clusters} ({num_treated} geo per treatment)")
    
    # MDE Constants
    _z_alpha = _norm.ppf(1 - 0.05 / 2)  # alpha=0.05
    _z_beta = _norm.ppf(0.80)            # power=80%
    TABLE_W = 160

    mde_col_label = f"MDE@{post_days}d" if post_len > 0 else "MDE"
    synth_label = "Matched" if is_did else "Synthetic"

    # ── Pre-compute all row values to determine dynamic column widths ──
    rows_data = []
    tot_lifts_abs = []
    tot_real_abs  = []
    tot_synth_abs = []
    ci_lowers_abs = []
    ci_uppers_abs = []

    for i, res in enumerate(results):
        treatment_str = ", ".join(res["cluster"]["treatment"])

        syn = res["synthetic"]
        post_synth_sum = syn["plotting_data"]["post_synth"].sum()
        post_real_sum  = syn["plotting_data"]["post_real"].sum()
        lift_abs = syn["lift_total"]
        lift_pct = lift_abs / post_synth_sum if post_synth_sum != 0 else 0

        ci_l_pct = syn["bootstrap"]["ci_lower_total_pct"]
        ci_u_pct = syn["bootstrap"]["ci_upper_total_pct"]
        ci_l_abs = syn["bootstrap"]["ci_lower_total_abs"]
        ci_u_abs = syn["bootstrap"]["ci_upper_total_abs"]

        sig_flag    = (ci_l_abs > 0 or ci_u_abs < 0)
        placebo_p   = res["placebo"]["p_value"]
        causal_flag = placebo_p <= 0.10

        # MDE calculation — matching DoE methodology (log-diff regression with weights)
        df_syn     = syn["df"]
        t_idx      = syn["plotting_data"]["treatment_idx"]
        treat_geos = res["cluster"]["treatment"]
        ctrl_geos  = list(syn["weights"].keys())

        try:
            pre_data = df_syn.iloc[:t_idx].copy()
            if len(treat_geos) > 1:
                treat_series = pre_data[treat_geos].mean(axis=1)
            else:
                treat_series = pre_data[treat_geos[0]]

            cols_for_log = pd.DataFrame({"_treat": treat_series.values}, index=pre_data.index)
            for g in ctrl_geos:
                cols_for_log[g] = pre_data[g].values

            log_diff = np.log(cols_for_log).diff().dropna()
            y_ld     = log_diff["_treat"].values
            X_ld     = log_diff[ctrl_geos].values
            w_vals   = np.array([syn["weights"][g] for g in ctrl_geos])
            y_pred_ld = X_ld @ w_vals
            sigma    = (y_ld - y_pred_ld).std()

            n_days      = post_days if post_len > 0 else 21
            delta_log   = (_z_alpha + _z_beta) * sigma / np.sqrt(n_days)
            cluster_mde = np.exp(delta_log) - 1
            mde_str     = f"{cluster_mde*100:.2f}%"
        except Exception:
            mde_str = "N/A"

        lift_pct_str = f"{lift_pct*100:.2f}%"
        ci_str       = f"[{ci_l_pct*100:.2f}%, {ci_u_pct*100:.2f}%]"
        ci_abs_str   = f"[{ci_l_abs:.1f}, {ci_u_abs:.1f}]"
        obs_str      = f"{post_real_sum:.2f}"
        syn_str      = f"{post_synth_sum:.2f}"
        abs_str      = f"{lift_abs:.2f}"
        sig_str      = "[Yes]" if sig_flag   else "[No]"
        cau_str      = "[Yes]" if causal_flag else "[No]"

        rows_data.append((i, treatment_str, obs_str, syn_str, lift_pct_str,
                          abs_str, ci_str, ci_abs_str, sig_str, cau_str, mde_str))

        tot_lifts_abs.append(lift_abs)
        tot_real_abs.append(post_real_sum)
        tot_synth_abs.append(post_synth_sum)
        ci_lowers_abs.append(ci_l_abs)
        ci_uppers_abs.append(ci_u_abs)

    # ── Dynamic widths ──
    treat_w   = max(len("Treatment"),  max(len(r[1])  for r in rows_data))
    obs_w     = max(len("Observed"),   max(len(r[2])  for r in rows_data))
    syn_w     = max(len(synth_label),  max(len(r[3])  for r in rows_data))
    lpct_w    = max(len("Lift (%)"),   max(len(r[4])  for r in rows_data))
    labs_w    = max(len("Lift (abs)"), max(len(r[5])  for r in rows_data))
    cipct_w   = max(len("CI 95% (%)"),max(len(r[6])  for r in rows_data))
    ciabs_w   = max(len("CI 95% (abs)"), max(len(r[7]) for r in rows_data))
    sig_w     = max(len("Sig"),    5)
    cau_w     = max(len("Causal"), 6)
    mde_w     = max(len(mde_col_label), max(len(r[10]) for r in rows_data))

    sep = "| "
    header = (f"{'Cluster':<7}{sep}{'Treatment':<{treat_w}}{sep}"
              f"{'Observed':<{obs_w}}{sep}{synth_label:<{syn_w}}{sep}"
              f"{'Lift (%)':<{lpct_w}}{sep}{'Lift (abs)':<{labs_w}}{sep}"
              f"{'CI 95% (%)':<{cipct_w}}{sep}{'CI 95% (abs)':<{ciabs_w}}{sep}"
              f"{'Sig':<{sig_w}}{sep}{'Causal':<{cau_w}}{sep}{mde_col_label}")
    dyn_w = len(header) + 2
    TABLE_W = max(TABLE_W, dyn_w)

    print("\n" + "-" * TABLE_W)
    print(" CLUSTER-LEVEL INCREMENTAL IMPACT ".center(TABLE_W, "-"))
    print("-" * TABLE_W)
    print(header)
    print("-" * TABLE_W)

    for (i, treatment_str, obs_str, syn_str, lift_pct_str,
         abs_str, ci_str, ci_abs_str, sig_str, cau_str, mde_str) in rows_data:
        row = (f"{i:<7}{sep}{treatment_str:<{treat_w}}{sep}"
               f"{obs_str:<{obs_w}}{sep}{syn_str:<{syn_w}}{sep}"
               f"{lift_pct_str:<{lpct_w}}{sep}{abs_str:<{labs_w}}{sep}"
               f"{ci_str:<{cipct_w}}{sep}{ci_abs_str:<{ciabs_w}}{sep}"
               f"{sig_str:<{sig_w}}{sep}{cau_str:<{cau_w}}{sep}{mde_str}")
        print(row)

    print("-" * TABLE_W)
    
    sum_real = sum(tot_real_abs)
    sum_synth = sum(tot_synth_abs)
    sum_lift = sum_real - sum_synth
    agg_lift_pct = sum_lift / sum_synth if sum_synth != 0 else 0.0
    
    # Proper aggregated bootstrap for confidence intervals
    try:
        from reallift.geo.bootstrap import bootstrap_significance
        
        # Extract time series arrays for the post period
        post_len = len(results[0]["synthetic"]["plotting_data"]["post_real"])
        consolidated_post_real = np.zeros(post_len)
        consolidated_post_synth = np.zeros(post_len)
        
        for res in results:
            consolidated_post_real += res["synthetic"]["plotting_data"]["post_real"]
            consolidated_post_synth += res["synthetic"]["plotting_data"]["post_synth"]
            
        consolidated_effect = consolidated_post_real - consolidated_post_synth
        cons_boot = bootstrap_significance(consolidated_effect, consolidated_post_synth, random_state=random_state)
        
        agg_ci_l_abs = cons_boot["ci_lower_total_abs"]
        agg_ci_u_abs = cons_boot["ci_upper_total_abs"]
        agg_ci_l_pct = cons_boot["ci_lower_total_pct"]
        agg_ci_u_pct = cons_boot["ci_upper_total_pct"]
    except Exception:
        # Fallback to sum of bounds
        agg_ci_l_abs = sum(ci_lowers_abs)
        agg_ci_u_abs = sum(ci_uppers_abs)
        agg_ci_l_pct = agg_ci_l_abs / sum_synth if sum_synth != 0 else 0.0
        agg_ci_u_pct = agg_ci_u_abs / sum_synth if sum_synth != 0 else 0.0
    
    # Consolidated MDE — average of per-cluster residuals (matching DoE consolidated mode)
    try:
        all_cluster_residuals = []
        for res in results:
            syn = res["synthetic"]
            t_idx = syn["plotting_data"]["treatment_idx"]
            treat_geos = res["cluster"]["treatment"]
            ctrl_geos = list(syn["weights"].keys())
            df_syn = syn["df"]
            pre_data = df_syn.iloc[:t_idx].copy()
            
            if len(treat_geos) > 1:
                treat_series = pre_data[treat_geos].mean(axis=1)
            else:
                treat_series = pre_data[treat_geos[0]]
            
            cols_for_log = pd.DataFrame({"_treat": treat_series.values}, index=pre_data.index)
            for g in ctrl_geos:
                cols_for_log[g] = pre_data[g].values
            
            log_diff = np.log(cols_for_log).diff().dropna()
            y_ld = log_diff["_treat"].values
            X_ld = log_diff[ctrl_geos].values
            w_vals = np.array([syn["weights"][g] for g in ctrl_geos])
            residuals = y_ld - X_ld @ w_vals
            all_cluster_residuals.append(pd.Series(residuals))
        
        # Average per-cluster residuals → std of the mean (same as DoE consolidated)
        residuals_df = pd.concat([r.reset_index(drop=True) for r in all_cluster_residuals], axis=1)
        mean_residuals = residuals_df.mean(axis=1)
        sigma_cons = mean_residuals.std()
        
        n_days_cons = post_days if post_len > 0 else 21
        delta_cons = (_z_alpha + _z_beta) * sigma_cons / np.sqrt(n_days_cons)
        consolidated_mde = np.exp(delta_cons) - 1
        cons_mde_str = f"{consolidated_mde*100:.2f}%"
    except Exception:
        cons_mde_str = "N/A"
    
    print(f"\n=== CONSOLIDATED IMPACT ({mde_col_label}: {cons_mde_str}) ===\n")
    expected_label = "Total Matched Baseline (Expected)" if is_did else "Total Synthetic (Expected)"
    print(f"  {'Total Observed Output':<33}: {sum_real:,.2f}")
    print(f"  {expected_label:<33}: {sum_synth:,.2f}")
    print(f"  --------------------------------------------------")
    print(f"  {'INCREMENTAL ABSOLUTE LIFT':<33}: {sum_lift:,.2f}")
    print(f"  {'95% Confidence Interval (abs)':<33}: [{agg_ci_l_abs:,.1f}, {agg_ci_u_abs:,.1f}]")
    print(f"  --------------------------------------------------")
    print(f"  {'INCREMENTAL PERCENTUAL LIFT':<33}: {agg_lift_pct*100:.2f}%")
    print(f"  {'95% Confidence Interval (%)':<33}: [{agg_ci_l_pct*100:.2f}%, {agg_ci_u_pct*100:.2f}%]")
    
    final_sig = "[Yes] Statistically Significant" if (agg_ci_l_abs > 0 or agg_ci_u_abs < 0) else "[No] Not Statistically Significant"
    print(f"\n  Result: {final_sig}\n")
    print("=" * 70 + "\n")

def _print_scenario_table(clusters, duration, mde, cv_summary=None, total_geos=None, experiment_days=None, experiment_type=None):
    """Print compact per-cluster table for a scenario with CV metrics and donor pools."""
    cluster_results = duration["cluster_results"]
    consolidated = duration["consolidated"]

    is_auto = mde is None

    # ── Duration & MDE Grid ──
    if is_auto:
        days_to_print = experiment_days if experiment_days else [21, 30, 60]
        mde_cols = [f"MDE @{d}d" for d in days_to_print]
        mde_hdr = " | ".join(f"{col:<9}" for col in mde_cols)

        # Dynamic column width based on longest treatment name
        treat_w = max((len(", ".join(cl["treatment"])) for cl in clusters), default=10)
        treat_w = max(treat_w, len("Treatment"))

        print(f"\n  {'Cluster':<7} | {'Treatment':<{treat_w}} | {'Controls':<8} | {mde_hdr}")
        print("  " + "-" * (20 + treat_w + len(mde_hdr)))

        for i, (cl, cr) in enumerate(zip(clusters, cluster_results)):
            treat = ", ".join(cl["treatment"])
            weights = cl.get("control_weights", [])
            n_ctrl = sum(1 for w in weights if w > 0.001) if weights else len(cl["control"])
            
            curve = cr.get("mde_curve")

            mde_strs = []
            for d in days_to_print:
                if curve is not None:
                    val = curve.loc[curve["days"] == d, "mde"]
                    mde_strs.append(f"{val.values[0]*100:.2f}%" if len(val) > 0 else "N/A")
                else:
                    mde_strs.append("N/A")

            mde_row = " | ".join(f"{s:<9}" for s in mde_strs)
            print(f"  {i:<7} | {treat:<{treat_w}} | {n_ctrl:<8} | {mde_row}")

        # Consolidated row
        c_curve = consolidated.get("mde_curve")
        c_mdes = []
        for d in days_to_print:
            if c_curve is not None:
                val = c_curve.loc[c_curve["days"] == d, "mde"]
                c_mdes.append(f"{val.values[0]*100:.2f}%" if len(val) > 0 else "N/A")
            else:
                c_mdes.append("N/A")

        c_mdes_str = " | ".join(f"{s:<9}" for s in c_mdes)
        print("  " + "-" * (20 + treat_w + len(mde_hdr)))
        print(f"  {'CONSOL.':<7} | {'pooled':<{treat_w}} | {'':<8} | {c_mdes_str}")
    else:
        # Dynamic column width based on longest treatment name
        treat_w = max((len(", ".join(cl["treatment"])) for cl in clusters), default=10)
        treat_w = max(treat_w, len("Treatment"))

        print(f"\n  {'Cluster':<7} | {'Treatment':<{treat_w}} | {'Controls':<8} | {'Min Days':<8} | {'Power':<7}")
        print("  " + "-" * (36 + treat_w))

        for i, (cl, cr) in enumerate(zip(clusters, cluster_results)):
            treat = ", ".join(cl["treatment"])
            weights = cl.get("control_weights", [])
            n_ctrl = sum(1 for w in weights if w > 0.001) if weights else len(cl["control"])

            best_days = cr["summary"].get("best_days")
            best_power = cr["summary"].get("best_power")

            days_str = f"{int(best_days)}d" if best_days else "N/A"
            power_str = f"{best_power:.1%}" if best_power else "N/A"

            print(f"  {i:<7} | {treat:<{treat_w}} | {n_ctrl:<8} | {days_str:<8} | {power_str:<7}")

        c_summary = consolidated["summary"]
        c_best = c_summary.get("best_days")
        c_power = c_summary.get("best_power")
        c_days_str = f"{int(c_best)}d" if c_best else "N/A"
        c_power_str = f"{c_power:.1%}" if c_power else "N/A"

        print("  " + "-" * (36 + treat_w))
        print(f"  {'CONSOL.':<7} | {'pooled':<{treat_w}} | {'':<8} | {c_days_str:<8} | {c_power_str:<7}")

    # ── Treatment & Donor Layout ──
    all_treatments = set()
    all_controls = set()
    for cl in clusters:
        all_treatments.update(cl["treatment"])
        all_controls.update(cl["control"])
    
    distinct_geos = len(all_treatments | all_controls)
    
    print("\n  EXPERIMENTAL SCOPE")
    coverage_str = f"{distinct_geos / total_geos:.0%}" if total_geos else "N/A"
    print(f"  Distinct Geos Used   : {distinct_geos} ({coverage_str} coverage)")
    
    print(f"\n  TEST POOL (TREATMENT UNITS): {', '.join(sorted(list(all_treatments)))}")
    
    print(f"\n  CONTROL DESIGN (DONOR POOLS)")
    for i, cl in enumerate(clusters):
        treat_name = ", ".join(cl["treatment"])
        controls = cl["control"]
        weights = cl.get("control_weights", [])
        
        # Filter to significant donors (weight > 0.001)
        donors = [(d, w) for d, w in zip(controls, weights) if w > 0.001]
        donors.sort(key=lambda x: x[1], reverse=True)
        n_donors = len(donors)
        
        print(f"\n  Cluster {i} — {treat_name} ({n_donors} donor{'s' if n_donors != 1 else ''})")
        print(f"  {'─' * 50}")
        
        # Print donors in two columns for compactness
        for j in range(0, len(donors), 2):
            left = f"  {donors[j][0]:<22} {donors[j][1]:.3f}"
            if j + 1 < len(donors):
                right = f"  {donors[j+1][0]:<22} {donors[j+1][1]:.3f}"
            else:
                right = ""
            print(f"{left}{right}")

    # ── Cross-Validation Grid ──
    warnings_list = []  # Collect warnings for dedicated section
    if cv_summary is not None and not cv_summary.empty:
        is_pure_did = experiment_type == "matched_did"
        
        # Dynamic column width for Treatment
        treat_w = max(
            (len(", ".join(row["treatment"]) if isinstance(row["treatment"], list) else str(row["treatment"]))
             for _, row in cv_summary.iterrows()),
            default=10
        )
        treat_w = max(treat_w, len("Treatment"))

        if is_pure_did:
            print(f"\n  CROSS-VALIDATION SUMMARY")
            print(f"  {'Cluster':<7} | {'Treatment':<{treat_w}} | {'R²':<17}")
            print("  " + "-" * (20 + treat_w))

            for i, row in cv_summary.iterrows():
                treat = ", ".join(row["treatment"]) if isinstance(row["treatment"], list) else str(row["treatment"])
                r2 = f"{row['r2_test']:.4f}"
                print(f"  {i:<7} | {treat:<{treat_w}} | {r2:<17}")
        else:
            print(f"\n  CROSS-VALIDATION SUMMARY")
            print(f"  {'Cluster':<7} | {'Treatment':<{treat_w}} | {'R² Train':<8} | {'R² Test':<8} | {'MAPE Tr':<8} | {'MAPE Te':<8} | {'WAPE Tr':<8} | {'WAPE Te':<8}")
            print("  " + "-" * (47 + treat_w))

            for i, row in cv_summary.iterrows():
                treat = ", ".join(row["treatment"]) if isinstance(row["treatment"], list) else str(row["treatment"])

                r2_tr = f"{row['r2_train']:.4f}"
                r2_te = f"{row['r2_test']:.4f}"
                mape_tr = f"{row['mape_train']:.4f}"
                mape_te = f"{row['mape_test']:.4f}"
                wape_tr = f"{row['wape_train']:.4f}"
                wape_te = f"{row['wape_test']:.4f}"

                # Track warnings (don't print inline)
                gap = row['r2_train'] - row['r2_test']
                if row['r2_test'] < 0.60:
                    warnings_list.append(f"  {treat:<{treat_w}} R² test = {row['r2_test']:.4f} (below 0.60 threshold)")
                elif row['r2_train'] < 0.60:
                    warnings_list.append(f"  {treat:<{treat_w}} R² train = {row['r2_train']:.4f} (below 0.60 threshold)")
                elif abs(gap) > 0.2:
                    warnings_list.append(f"  {treat:<{treat_w}} R² gap = {gap:.4f} (instability risk)")

                print(f"  {i:<7} | {treat:<{treat_w}} | {r2_tr:<8} | {r2_te:<8} | {mape_tr:<8} | {mape_te:<8} | {wape_tr:<8} | {wape_te:<8}")

    
    # ── Dedicated WARNINGS Section ──
    if warnings_list or (experiment_type == "synthetic_control" and any(
        r.get("r2_test", 1.0) < 0.60 if isinstance(r, dict) else (r["r2_test"] < 0.60 if "r2_test" in r.index else False)
        for r in (cv_summary.iloc[i] for i in range(len(cv_summary))) if cv_summary is not None
    )):
        print(f"\n  WARNINGS")
        print(f"  {'─' * 70}")
        if warnings_list:
            n_warn = len(warnings_list)
            print(f"  [Warning] {n_warn} cluster{'s' if n_warn > 1 else ''} with quality concerns:")
            for w in warnings_list:
                print(f"            -{w}")
        
        # Check if any cluster has low R² and suggest alternatives
        has_low_quality = any(
            row["r2_test"] < 0.60
            for _, row in cv_summary.iterrows()
        ) if cv_summary is not None and not cv_summary.empty else False
        
        if has_low_quality:
            n_clusters = len(clusters) if clusters else 0
            print(f"  [Warning] Data may be too volatile for {n_clusters} simultaneous Synthetic")
            print(f"            Control clusters. Consider alternatives:")
            print(f"            - Try search_mode='exhaustive' for optimal partitioning")
            print(f"            - Reduce number of treatment geos")
            print(f"            - Use experiment_type='matched_did'")
        print(f"  {'─' * 70}")

def _build_comparison(scenarios, mde, experiment_days):
    """Build summary table comparing multiple scenarios."""
    rows = []
    for s_idx, s in enumerate(scenarios):
        if s.get("clusters") is None:
            continue
        
        pct = s.get("pct_treatment", 0)
        n_treat = s["n_treatment"]
        consolidated = s["duration"]["consolidated"]["summary"]
        
        # Calculate distinct geos
        all_treat = set()
        all_ctrl = set()
        for cl in s["clusters"]:
            all_treat.update(cl["treatment"])
            all_ctrl.update(cl["control"])
        distinct_geos = len(all_treat | all_ctrl)
        n_controls = len(all_ctrl)

        row = {
            "Scenario": s_idx + 1,
            "% Treated": f"{pct:.0%}",
            "Clusters": len(s["clusters"]),
            "Distinct": distinct_geos,
            "Controls": n_controls,
            "sigma": consolidated["sigma"]
        }

        if mde is None:
            # Auto-MDE mode
            curve = s["duration"]["consolidated"].get("mde_curve")
            days_to_eval = experiment_days if experiment_days else [21, 30, 60]
            for d in days_to_eval:
                if curve is not None:
                    val = curve.loc[curve["days"] == d, "mde"]
                    row[f"mde_{d}d"] = f"{val.values[0]*100:.2f}%" if len(val) > 0 else "N/A"
                else:
                    row[f"mde_{d}d"] = "N/A"
        else:
            # Fixed MDE mode
            best_days = consolidated.get("best_days")
            best_power = consolidated.get("best_power")
            row["best_days"] = f"{int(best_days)}d" if best_days else "N/A"
            row["best_power"] = f"{best_power:.1%}" if best_power else "N/A"

        rows.append(row)

    return pd.DataFrame(rows)

def _print_comparison_table(comparison_df, mde, experiment_days=None):
    """Print the final consolidated comparison cross-scenario."""
    if comparison_df.empty:
        print("\n  No valid scenarios found to compare.")
        return

    print("\n" + "=" * 85)
    print(" EXPERIMENT DESIGN COMPARISON ".center(85, "="))
    print("=" * 85)
    print("")
    
    is_auto = mde is None
    if is_auto:
        days_to_print = experiment_days if experiment_days else [21, 30, 60]
        mde_cols = [f"MDE @{d}d" for d in days_to_print]
        mde_hdr = " | ".join(f"{col:<9}" for col in mde_cols)
        
        header = f"  {'Scenario':<8} | {'% Treated':<10} | {'Clusters':<8} | {'Distinct':<8} | {'Controls':<8} | {mde_hdr}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in comparison_df.iterrows():
            mde_vals = [f"{row.get(f'mde_{d}d', 'N/A'):<9}" for d in days_to_print]
            mde_row = " | ".join(mde_vals)
            print(f"  {int(row['Scenario']):<8} | {row['% Treated']:<10} | {int(row['Clusters']):<8} | {int(row['Distinct']):<8} | {int(row['Controls']):<8} | {mde_row}")
    else:
        header = f"  {'Scenario':<8} | {'% Treated':<10} | {'Clusters':<8} | {'Distinct':<8} | {'Controls':<8} | {'Min Days':<10} | {'Power':<7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in comparison_df.iterrows():
            print(f"  {int(row['Scenario']):<8} | {row['% Treated']:<10} | {int(row['Clusters']):<8} | {int(row['Distinct']):<8} | {int(row['Controls']):<8} | {row['best_days']:<10} | {row['best_power']:<7}")
    
    print("\n" + "=" * 85)
