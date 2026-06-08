import pandas as pd
import numpy as np
import warnings
from ._discovery import discover_geo_clusters
from ._duration import estimate_duration
from ..config.defaults import DEFAULT_TREATMENT_PCTS, DEFAULT_EXPERIMENT_DAYS
from ._shared import _run_oof_refinement_single
from ._reporting import _print_scenario_table, _build_comparison, _print_comparison_table
from ._reporting import generate_doe_report
import cvxpy as cp

def _check_ghost_lift_oos(clusters, df_pre, date_col, experiment_days):
    """
    Per-cluster ghost lift detection using CMBB.

    For each cluster × experiment duration d:
      - Holdout the last d days of pre-period (pseudo-intervention, true lift = 0)
      - Fit SCM on the remaining pre-period (train)
      - Apply weights to the holdout → compute residuals
      - Run CMBB (bootstrap_significance) on those residuals
      - Ghost lift detected if the 95% CI does NOT cross zero

    Checks each cluster independently — a consolidated pass cannot hide a
    per-cluster failure (unlike the old aggregated approach).

    Returns True if ANY cluster shows ghost lift for ANY requested duration.
    """
    from ._bootstrap import bootstrap_significance

    if not isinstance(clusters, list):
        clusters = [clusters]

    if not isinstance(experiment_days, (list, tuple)):
        eval_sizes = [int(experiment_days)]
    else:
        eval_sizes = sorted(int(d) for d in experiment_days)

    for cluster in clusters:
        treat_cols = cluster["treatment"]
        controls   = cluster["control"]
        if not controls:
            continue

        for d in eval_sizes:
            if d >= len(df_pre) - 5:
                continue

            train_df = df_pre.iloc[:-d]
            test_df  = df_pre.iloc[-d:]

            y_train = train_df[treat_cols].mean(axis=1).values.astype(float)
            X_train = train_df[controls].values.astype(float)

            y_mean = y_train.mean() or 1e-10
            X_mean = np.where(X_train.mean(axis=0) == 0, 1e-10, X_train.mean(axis=0))

            w_syn = cp.Variable(X_train.shape[1])
            prob  = cp.Problem(
                cp.Minimize(cp.sum_squares(y_train / y_mean - (X_train / X_mean) @ w_syn)),
                [w_syn >= 0, cp.sum(w_syn) == 1],
            )
            try:
                prob.solve(solver=cp.SCS, verbose=False)
                w_vals = np.clip(np.array(w_syn.value).flatten(), 0, None)
                s = w_vals.sum()
                if s > 0:
                    w_vals /= s
            except Exception:
                w_vals = np.ones(X_train.shape[1]) / X_train.shape[1]

            y_test    = test_df[treat_cols].mean(axis=1).values.astype(float)
            X_test    = test_df[controls].values.astype(float)
            synthetic = (X_test / X_mean) @ w_vals * y_mean

            effect          = y_test - synthetic
            post_synth_safe = np.where(synthetic <= 0, 1e-10, synthetic)

            try:
                bs = bootstrap_significance(
                    effect, post_synth_safe, n_boot=500, random_state=42
                )
                if bs["ci_lower_pct"] > 0 or bs["ci_upper_pct"] < 0:
                    return True  # spurious significant lift detected
            except Exception:
                continue

    return False


def _cluster_geos_by_scale(geos, df, k, exclude_outliers=True):
    """
    Cluster geos into k groups by pre-period mean using K-Means in log space.

    Log transform handles skewed (log-normal) geo distributions — clusters
    capture natural multiplicative scale bands rather than forcing equal geo
    counts.  Cluster IDs are assigned in ascending order of centroid mean.

    When ``exclude_outliers=True``, geos outside the Tukey fence (Q1 − 1.5·IQR,
    Q3 + 1.5·IQR) computed in log space are removed before clustering.  They are
    returned separately so callers can exclude them from the treatment candidate
    pool while still making them available as donors.

    Returns
    -------
    geo_cluster  : dict {geo -> cluster_id}  (outliers not included)
    cluster_geos : dict {cluster_id -> [geos]}
    outlier_geos : list of geos flagged as outliers (empty when exclude_outliers=False)
    """
    from sklearn.cluster import KMeans

    geo_means = {g: df[g].mean() for g in geos}
    log_vals = np.array([np.log(max(geo_means[g], 1e-10)) for g in geos])

    outlier_geos = []
    cluster_geos_input = list(geos)

    if exclude_outliers and len(geos) >= 4:
        q1, q3 = np.percentile(log_vals, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        cluster_geos_input = [g for g, lv in zip(geos, log_vals) if lo <= lv <= hi]
        outlier_geos = [g for g, lv in zip(geos, log_vals) if lv < lo or lv > hi]
        log_vals = np.array([np.log(max(geo_means[g], 1e-10)) for g in cluster_geos_input])

    k = min(k, len(cluster_geos_input))
    log_means = log_vals.reshape(-1, 1)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw_labels = km.fit_predict(log_means)

    # Remap labels so cluster 0 = smallest mean, cluster k-1 = largest mean
    centroid_order = np.argsort(km.cluster_centers_.flatten())
    remap = {old: new for new, old in enumerate(centroid_order)}

    geo_cluster = {g: remap[raw_labels[i]] for i, g in enumerate(cluster_geos_input)}
    cluster_geos = {cid: [] for cid in range(k)}
    for g, cid in geo_cluster.items():
        cluster_geos[cid].append(g)
    return geo_cluster, cluster_geos, outlier_geos



def _find_optimal_k(log_values, max_k=8, max_within_ratio=3.0, plot=True, verbose=True):
    """
    Automatic k selection for scale clustering.

    First enforces a minimum k so that no cluster spans more than
    ``max_within_ratio``× in raw scale (e.g. 3.0 means the largest geo in a
    cluster is at most 3× the smallest).  Then picks the best k ≥ min_k by
    maximising the Silhouette Score in log space.

    Parameters
    ----------
    log_values : np.ndarray
        Log-transformed geo pre-period means.
    max_k : int
        Upper bound for k search.
    max_within_ratio : float
        Maximum acceptable scale ratio within a cluster (default 3.0×).
    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    max_k = min(max_k, len(log_values) - 1)

    # Minimum k derived from log range and max acceptable within-cluster ratio
    log_range = log_values.max() - log_values.min()
    min_k = max(2, int(np.ceil(log_range / np.log(max_within_ratio))))
    min_k = min(min_k, max_k)  # can't exceed max_k

    ks = list(range(2, max_k + 1))
    X = log_values.reshape(-1, 1)

    wcss_vals, sil_scores = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        wcss_vals.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))

    # Pick best silhouette among k >= min_k
    eligible = [(i, s) for i, (k, s) in enumerate(zip(ks, sil_scores)) if k >= min_k]
    best_idx = max(eligible, key=lambda t: t[1])[0]
    suggested_k = ks[best_idx]

    result_df = pd.DataFrame({"k": ks, "wcss": wcss_vals, "silhouette": sil_scores})

    if verbose:
        print(f"\n── Auto K Selection ──────────────────────────────────────")
        print(f"  Log range: {log_range:.2f}  |  max within ratio: {max_within_ratio}×  |  min_k enforced: {min_k}")
        print(result_df.to_string(index=False))
        print(f"\n  → Suggested k = {suggested_k}  (silhouette = {sil_scores[best_idx]:.4f})")

    if plot:
        with plt.style.context("dark_background"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3))
            fig.patch.set_facecolor("black")

            for ax in (ax1, ax2):
                ax.set_facecolor("black")
                ax.grid(True, linestyle="--", alpha=0.15, color="white")
                ax.tick_params(colors="#CBD5E1", labelsize=9)
                for spine in ax.spines.values():
                    spine.set_color("#1E1E1E")

            ax1.plot(ks, sil_scores, marker="o", color="#06B6D4", linewidth=1.5)
            if min_k > 2:
                ax1.axvspan(1.5, min_k - 0.5, color="#334155", alpha=0.5,
                            label=f"k < {min_k} excluded")
            ax1.axvline(suggested_k, color="#EF4444", linestyle="--", linewidth=1.2,
                        label=f"Best k = {suggested_k}")
            ax1.set_xlabel("k", color="#CBD5E1")
            ax1.set_ylabel("Silhouette Score", color="#CBD5E1")
            ax1.set_title("Silhouette Score (higher = better)", color="white", fontweight="bold")
            ax1.legend(fontsize=8, facecolor="#111111", edgecolor="#333333", labelcolor="white",
                       loc="lower right")

            ax2.plot(ks, wcss_vals, marker="o", color="#8B5CF6", linewidth=1.5)
            ax2.axvline(suggested_k, color="#EF4444", linestyle="--", linewidth=1.2)
            ax2.set_xlabel("k", color="#CBD5E1")
            ax2.set_ylabel("Within-cluster SS (log space)", color="#CBD5E1")
            ax2.set_title("WCSS — reference only", color="white", fontweight="bold")

            plt.suptitle("Scale Cluster Selection", fontsize=12, color="white", fontweight="bold")
            plt.tight_layout()
            plt.show()

    return suggested_k, result_df


def compute_scale_clusters(
    scale_clusters=None,
    geos=None,
    start_date=None,
    end_date=None,
    date_col=None,
    max_k=None,
    plot=True,
    verbose=True,
    df=None,
    filepath=None,
    exclude_outliers=True,
):
    """
    Compute scale-based geo clusters. Called by pre_clustering() (with plot/verbose)
    and internally by design_of_experiments() (with plot=False, verbose=False).

    When ``scale_clusters=None`` (default), automatically finds the optimal k
    using the Elbow / Kneedle method on within-cluster sum of squares (WCSS).
    Pass an explicit integer to inspect a specific k.

    Parameters
    ----------
    scale_clusters : int or None
        Number of scale clusters.  ``None`` triggers automatic k selection.
    max_k : int or None
        Upper bound for k search when ``scale_clusters=None``.
        Defaults to ``min(8, n_geos // 3)`` — ensures at least 3 geos per
        cluster (1 treatment + 2 donors minimum).
    geos : list, optional
        Subset of geos to consider. Defaults to all non-date columns.
    start_date, end_date : str, optional
        Date window for computing means (YYYY-MM-DD).
    date_col : str
        Name of the date column.
    plot : bool
        Whether to render a strip chart of geo means coloured by cluster.
    verbose : bool
        Whether to print the summary tables.
    df : pd.DataFrame, optional
        Pre-loaded DataFrame.  If omitted, ``filepath`` must be provided.
    filepath : str, optional
        Path to CSV file.

    Returns
    -------
    dict with keys:
        ``geo_df``     — geo-level DataFrame (geo, cluster, mean, std, cv)
        ``cluster_df`` — cluster-level summary
        ``distance_df``— pairwise distance matrix between cluster centroids
    """
    import matplotlib.pyplot as plt

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

    if geos is None:
        geos = [c for c in df.columns if c != date_col]

    geo_means_arr = np.array([df[g].mean() for g in geos])

    # Auto max_k: at least 3 geos per cluster (1 treatment + 2 donors minimum)
    if max_k is None:
        max_k = min(8, len(geos) // 3)
    max_k = max(2, max_k)  # always test at least k=2

    elbow_df = None
    if scale_clusters is None:
        # Elbow runs in log space — consistent with log-space clustering
        log_means_arr = np.log(np.where(geo_means_arr > 0, geo_means_arr, 1e-10))
        scale_clusters, elbow_df = _find_optimal_k(
            log_means_arr, max_k=max_k, plot=plot, verbose=verbose
        )
        if verbose:
            print(f"\n  Using k = {scale_clusters} for cluster preview below.\n")

    geo_cluster, cluster_geos_map, outlier_geos = _cluster_geos_by_scale(
        geos, df, scale_clusters, exclude_outliers=exclude_outliers
    )

    if outlier_geos and verbose:
        print(f"  [Outliers excluded from clustering] ({len(outlier_geos)} geos): {outlier_geos}")
        print(f"  These geos fall outside the Tukey fence (1.5×IQR in log space) and are")
        print(f"  excluded from treatment candidacy. They remain available as donors.\n")

    # Geo-level stats (outliers shown with cluster = "outlier")
    geo_rows = []
    for g in geos:
        s = df[g]
        geo_rows.append({
            "geo":     g,
            "cluster": geo_cluster.get(g, "outlier"),
            "mean":    round(s.mean(), 2),
            "std":     round(s.std(), 2),
            "cv":      round(s.std() / s.mean(), 4) if s.mean() != 0 else np.nan,
            "min":     round(s.min(), 2),
            "max":     round(s.max(), 2),
        })
    geo_df = pd.DataFrame(geo_rows).sort_values(["cluster", "mean"]).reset_index(drop=True)

    # Cluster-level summary
    cluster_rows = []
    for cid in range(scale_clusters):
        cg = cluster_geos_map[cid]
        means = [df[g].mean() for g in cg]
        cluster_rows.append({
            "cluster":    cid,
            "n_geos":     len(cg),
            "mean":       round(np.mean(means), 2),
            "std_within": round(np.std(means), 2),
            "min_geo_mean": round(np.min(means), 2),
            "max_geo_mean": round(np.max(means), 2),
            "geos":       cg,
        })
    cluster_df = pd.DataFrame(cluster_rows)

    # Pairwise inter-cluster distances (absolute difference between centroids)
    centroids = cluster_df["mean"].values
    k = len(centroids)
    dist_matrix = np.abs(centroids[:, None] - centroids[None, :])
    distance_df = pd.DataFrame(
        dist_matrix,
        index=[f"C{i}" for i in range(k)],
        columns=[f"C{i}" for i in range(k)],
    ).round(2)

    if verbose:
        print("\n" + "=" * 60)
        print(" SCALE CLUSTER PREVIEW ".center(60, "="))
        print("=" * 60)
        print(f"\nGeos: {len(geos)}  |  Clusters: {scale_clusters}\n")

        print("── Cluster Summary ──────────────────────────────────────")
        display_cols = ["cluster", "n_geos", "mean", "std_within", "min_geo_mean", "max_geo_mean"]
        print(cluster_df[display_cols].to_string(index=False))

        print("\n── Inter-Cluster Distance (|centroid_i − centroid_j|) ───")
        print(distance_df.to_string())

        print("\n── Geo Detail ───────────────────────────────────────────")
        print(geo_df[["geo", "cluster", "mean", "std", "cv"]].to_string(index=False))
        print()

    if plot:
        PALETTE = ["#06B6D4", "#F59E0B", "#10B981", "#EF4444",
                   "#8B5CF6", "#3B82F6", "#F97316", "#EC4899",
                   "#14B8A6", "#A3E635"]

        with plt.style.context("dark_background"):
            fig, (ax, ax2) = plt.subplots(
                2, 1,
                figsize=(11, max(5, scale_clusters * 0.9) + 3),
            )
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            ax2.set_facecolor("black")

            rng_jitter = np.random.default_rng(42)

            for cid in range(scale_clusters):
                cg = cluster_geos_map[cid]
                means = np.array([df[g].mean() for g in cg])
                color = PALETTE[cid % len(PALETTE)]

                # Left: band strip plot
                ax.scatter(means, [cid] * len(cg),
                           color=color, s=90, zorder=3, alpha=0.9,
                           label=f"Cluster {cid}  (n={len(cg)}, μ={centroids[cid]:,.0f})")
                ax.axhline(cid, color=color, linewidth=0.3, alpha=0.25)

                # Right: scatter with vertical jitter
                jitter = rng_jitter.uniform(-0.38, 0.38, size=len(cg))
                ax2.scatter(means, jitter, color=color, s=30, alpha=0.75, zorder=3)

            # Left axis styling
            ax.set_yticks(range(scale_clusters))
            ax.set_yticklabels([f"C{i}" for i in range(scale_clusters)], color="#CBD5E1")
            ax.set_xlabel("Geo pre-period mean", color="#CBD5E1")
            ax.set_title(f"Scale Clusters ({scale_clusters} bands) — Geo Means",
                         color="white", fontweight="bold", fontsize=12)
            ax.tick_params(colors="#CBD5E1", labelsize=9)
            ax.grid(True, axis="x", linestyle="--", alpha=0.15, color="white")
            for spine in ax.spines.values():
                spine.set_color("#1E1E1E")
            ax.legend(loc="lower right", fontsize=8,
                      facecolor="#111111", edgecolor="#333333", labelcolor="white")

            # Right axis styling
            ax2.set_xlabel("Geo pre-period mean", color="#CBD5E1")
            ax2.set_title("Geo Distribution — Individual Points",
                          color="white", fontweight="bold", fontsize=12)
            ax2.set_yticks([])
            ax2.tick_params(colors="#CBD5E1", labelsize=9)
            ax2.grid(True, axis="x", linestyle="--", alpha=0.15, color="white")
            for spine in ax2.spines.values():
                spine.set_color("#1E1E1E")

            plt.tight_layout()
            plt.show()

    return {
        "geo_df": geo_df,
        "cluster_df": cluster_df,
        "distance_df": distance_df,
        "suggested_k": scale_clusters,
        "elbow_df": elbow_df,
    }


def _cluster_mean_corr(df, cluster_geos_map, date_col, start_date, end_date):
    """Mean off-diagonal Pearson correlation within each scale cluster."""
    mask = pd.Series([True] * len(df), index=df.index)
    if start_date is not None:
        mask &= df[date_col] >= pd.to_datetime(start_date)
    if end_date is not None:
        mask &= df[date_col] <= pd.to_datetime(end_date)
    df_p = df.loc[mask]
    result = {}
    for cid, geos in cluster_geos_map.items():
        if len(geos) < 2:
            result[cid] = 0.0
            continue
        corr = df_p[geos].corr().values
        n = len(geos)
        off = corr[~np.eye(n, dtype=bool)]
        result[cid] = float(np.mean(off))
    return result


def _oof_status(cv, passed, ghost, r2_threshold=0.6, gap_threshold=0.15, wape_threshold=0.20, wdist_threshold=None):
    """Return a fixed-width (9-char) status string for OOF log lines."""
    if ghost:
        return "GhostLift"
    if passed:
        return "Approved "
    r2_test  = cv.get("r2_test",  0.0)
    r2_train = cv.get("r2_train", 0.0)
    if r2_test < r2_threshold or r2_train < r2_threshold:
        return "Fail R²  "
    if abs(r2_train - r2_test) > gap_threshold:
        return "Fail Gap "
    if wape_threshold is not None and cv.get("wape_test", 0.0) > wape_threshold:
        return "Fail WAPE"
    if wdist_threshold is not None and cv.get("wdist_pct", 0.0) > wdist_threshold * 100:
        return "Fail Wdst"
    return "Failed   "


def _sequential_cluster_design(
    sorted_cids, cluster_corr_map, cluster_ranking, cluster_geos_map,
    geo_cluster, geos, filepath, date_col, df_pre, df, start_date, end_date,
    n_folds, experiment_days, experiment_type, check_ghost_lift,
    method, check_oof, verbose,
    r2_threshold=0.6, gap_threshold=0.15, wape_threshold=0.20, wdist_threshold=None,
):
    """
    Sequential per-cluster design for restrict_donors=False + scale_clusters.

    For each cluster (ordered by mean intra-cluster correlation, descending):
      Phase 1 — intra-cluster donors only.
      Phase 2 — global donor pool (excluding already-locked donors).
      Phase 3 — fallback: best evaluated candidate regardless of threshold.

    After each cluster is resolved, the treatment geo AND its donors are locked
    and unavailable to subsequent clusters.
    """
    from ._shared import _run_oof_refinement_single
    from ._discovery import discover_geo_clusters

    locked_treatments = set()
    locked_donors = set()
    locked_out = []       # approved + fallback clusters
    fallback_names = []

    def _try(candidate, donor_pool):
        if not donor_pool:
            return None, None, False
        try:
            ceval = discover_geo_clusters(
                filepath=filepath, date_col=date_col, df=df,
                geos=donor_pool, fixed_treatment=[candidate],
                start_date=start_date, end_date=end_date,
                method=method, check_oof=check_oof,
                verbose=False, show_results=False,
            )
            cl = ceval[0].copy()
        except Exception:
            return None, None, False
        best, cv, passed, _ = _run_oof_refinement_single(
            cl, filepath, date_col, df_pre, start_date, end_date,
            n_folds, experiment_days=experiment_days,
            experiment_type=experiment_type, df=df,
            r2_threshold=r2_threshold, gap_threshold=gap_threshold,
            wape_threshold=wape_threshold, wdist_threshold=wdist_threshold,
        )
        return best, cv, passed

    def _no_ghost(best):
        if not check_ghost_lift or experiment_type == "matched_did":
            return True
        trial = [item["best"] for item in locked_out] + [best]
        return not _check_ghost_lift_oos(trial, df_pre, date_col, experiment_days)

    for cid in sorted_cids:
        mean_corr = cluster_corr_map.get(cid, 0.0)
        candidates = [c for c in cluster_ranking.get(cid, []) if c not in locked_treatments]

        if verbose:
            print(f"\n  C{cid}  μ_corr={mean_corr:.3f}  ({len(candidates)} candidates)", flush=True)

        selected = None
        tried = []   # (candidate, best, cv) for fallback pool

        # ── Phase 1: intra-cluster donors ────────────────────────────────────
        for candidate in candidates:
            if candidate in locked_donors:
                continue
            intra = [g for g in cluster_geos_map[cid]
                     if g != candidate
                     and g not in locked_treatments
                     and g not in locked_donors]
            best, cv, passed = _try(candidate, intra)
            if cv is None:
                continue
            tried.append((candidate, best, cv))
            r2, gap = cv["r2_test"], cv["r2_train"] - cv["r2_test"]
            ghost = passed and not _no_ghost(best)
            if verbose:
                st = _oof_status(cv, passed, ghost, r2_threshold, gap_threshold, wape_threshold, wdist_threshold)
                print(f"    [P1] {candidate:<22}  {st}  R²={r2:.3f}  Gap={gap:.3f}  WAPE={cv.get('wape_test', 0.0):.3f}  Wdst={cv.get('wdist_pct', 0.0):.1f}%", flush=True)
            if passed and not ghost:
                selected = (candidate, best, cv)
                break

        # ── Phase 2: global donors (minus locked donors) ──────────────────────
        if selected is None:
            if verbose:
                print(f"    → Phase 2: global donors", flush=True)
            for candidate in candidates:
                pool = [g for g in geos
                        if g != candidate
                        and g not in locked_treatments
                        and g not in locked_donors]
                best, cv, passed = _try(candidate, pool)
                if cv is None:
                    continue
                tried.append((candidate, best, cv))
                r2, gap = cv["r2_test"], cv["r2_train"] - cv["r2_test"]
                ghost = passed and not _no_ghost(best)
                if verbose:
                    st = _oof_status(cv, passed, ghost, r2_threshold, gap_threshold, wape_threshold, wdist_threshold)
                    print(f"    [P2] {candidate:<22}  {st}  R²={r2:.3f}  Gap={gap:.3f}  WAPE={cv.get('wape_test', 0.0):.3f}  Wdst={cv.get('wdist_pct', 0.0):.1f}%", flush=True)
                if passed and not ghost:
                    selected = (candidate, best, cv)
                    break

        # ── Phase 3: fallback — best evaluated candidate ──────────────────────
        if selected is None:
            if verbose:
                print(f"    → Fallback", flush=True)
            if tried:
                best_fb = max(tried, key=lambda x: x[2]["r2_test"])
                selected = best_fb
                fallback_names.append(f"C{cid}:{best_fb[0]}")
                if verbose:
                    print(f"    [FB] {best_fb[0]:<22}  Fallback  R²={best_fb[2]['r2_test']:.3f}", flush=True)
            else:
                if verbose:
                    print(f"    [!!] C{cid}: no candidates evaluated — slot empty.", flush=True)
                continue

        # ── Lock ──────────────────────────────────────────────────────────────
        candidate, best, cv = selected
        locked_treatments.add(candidate)
        locked_donors.update(best.get("control", []))
        locked_out.append({"candidate": candidate, "best": best, "cv": cv})
        if verbose:
            n_d = len(best.get("control", []))
            print(f"    → Locked: {candidate} ({n_d} donors)", flush=True)

    refined = [
        {"raw": None, "best": r["best"], "cv": r["cv"],
         "scale_cluster_id": geo_cluster[r["candidate"]]}
        for r in locked_out
    ]
    return refined, fallback_names


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
    method="penalized_scm",
    elasticnet_alpha=0.01,
    elasticnet_l1_ratio=0.5,
    check_ghost_lift=True,
    check_oof=True,
    r2_threshold=0.6,
    gap_threshold=0.15,
    wape_threshold=0.20,
    wdist_threshold=None,
    n_jobs=None,
    verbose=True,
    save_pdf=False,
    pdf_name='doe_report.pdf',
    logo=None,
    use_bootstrap_mde=True,
    scale_clusters=None,
    restrict_donors=False,
    exclude_outliers=True,
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
    elif scale_clusters is not None:
        # Scale-cluster mode: n_treatment is determined by k, not pct_treatment.
        # A single scenario is sufficient — diversity across market scales is
        # already guaranteed by the stratified selection (one geo per cluster).
        if pct_treatment is not None and verbose:
            print("  [Info] scale_clusters is set — pct_treatment is ignored. "
                  "Running a single stratified scenario.\n")
        scenario_configs = [{
            "pct": scale_clusters / n_geos,
            "n_treatment": scale_clusters,
            "fixed": None,
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
            print(f"  [Info] Ghost Lift check enabled — CMBB per-cluster for each requested duration.\n")

    # 3. Global Base Ranking
    max_n_treat = max(c["n_treatment"] for c in scenario_configs)
    has_fixed = any(c["fixed"] for c in scenario_configs)

    global_ranking = None
    _skip_global_rank = (scale_clusters is not None and restrict_donors and not has_fixed)
    from ._discovery import _evaluate_combinations

    # df_rank is needed for both global ranking and per-cluster restricted ranking
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
    df_rank = df_rank.copy()
    df_rank = df_rank.sort_values(date_col).reset_index(drop=True)

    if not has_fixed and search_mode == "ranking" and not _skip_global_rank:
        if verbose: print("\n  Evaluating full historical data for global ranking...")
        valid_combos = [[g] for g in geos]
        phase1_results_full = _evaluate_combinations(
            df_rank, geos, valid_combos,
            all_treatment_geos=set(),
            method=method,
            alpha_grid=[elasticnet_alpha], l1_grid=[elasticnet_l1_ratio],
            verbose=verbose, desc="Ranking Geos", n_jobs=n_jobs,
            check_oof=check_oof,
        )
        phase1_results_full.sort(key=lambda x: x["ser"])
        global_ranking = [r["treatment"][0] for r in phase1_results_full]

        if verbose:
            print(f"  Global ranking (top {max_n_treat}): {global_ranking[:max_n_treat]}\n")

    # Scale clustering (optional)
    geo_cluster, cluster_geos_map = None, None
    if scale_clusters is not None and not has_fixed:
        _sc_result = compute_scale_clusters(
            scale_clusters=scale_clusters,
            geos=geos,
            start_date=start_date,
            end_date=end_date,
            date_col=date_col,
            plot=False,
            verbose=False,
            df=df,
            exclude_outliers=exclude_outliers,
        )
        _geo_df = _sc_result["geo_df"]
        _non_outlier = _geo_df[_geo_df["cluster"] != "outlier"]
        geo_cluster = dict(zip(_non_outlier["geo"], _non_outlier["cluster"]))
        cluster_geos_map = {}
        for _geo, _cid in geo_cluster.items():
            cluster_geos_map.setdefault(_cid, []).append(_geo)
        outlier_geos = _geo_df[_geo_df["cluster"] == "outlier"]["geo"].tolist()
        if outlier_geos and verbose:
            print(f"  [Outliers excluded] ({len(outlier_geos)} geos): {outlier_geos}")
            print(f"  Excluded from treatment candidacy; still available as donors.\n")
        if verbose:
            print(f"  Scale clusters ({scale_clusters}): ", end="")
            for cid, cg in cluster_geos_map.items():
                means = [round(df[g].mean(), 1) for g in cg]
                print(f"\n    Cluster {cid} ({len(cg)} geos, mean ≈ {round(sum(means)/len(means),1)}): {cg}", end="")
            print("\n")

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
            elif global_ranking is not None or (search_mode == "ranking" and not config["fixed"]):
                # Greedy Sequential Search mode handles discovery inside OOF loop.
                # Also enters here when global_ranking was skipped (restrict_donors + scale_clusters):
                # per-cluster rankings are built directly inside the greedy search.
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
                    method=method,
                    search_mode=search_mode,
                    check_oof=check_oof,
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

        if search_mode_used == "ranking":
            if run_as_matched_did:
                if verbose: print("\n  [Matched DiD] Evaluating all candidates to maximize correlation...")
                
                all_evals = []
                for candidate in geos:
                    try:
                        candidate_eval = discover_geo_clusters(
                            filepath=filepath, date_col=date_col, geos=geos,
                            fixed_treatment=[candidate], start_date=start_date, end_date=end_date,
                            method=method, check_oof=check_oof,
                            verbose=False, show_results=False, df=df
                        )
                        current_cluster = candidate_eval[0].copy()
                    except Exception:
                        continue
                    
                    best_cluster, best_cv_row, passed, iters = _run_oof_refinement_single(
                        current_cluster, filepath, date_col, df_pre, start_date, end_date, n_folds,
                        experiment_days=experiment_days, experiment_type=experiment_type, df=df,
                        r2_threshold=r2_threshold, gap_threshold=gap_threshold,
                        wape_threshold=wape_threshold, wdist_threshold=wdist_threshold,
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
                donors_mode = "restricted" if restrict_donors else "global"
                if verbose:
                    print(f"\n  [Greedy Lock Search]  (donors: {donors_mode})\n")

                # Build per-cluster ranked lists (scale clustering)
                if geo_cluster is not None:
                    cluster_ranking = {}

                    if restrict_donors and verbose:
                        parts = "  ".join(f"C{cid}({len(cluster_geos_map[cid])})" for cid in range(scale_clusters))
                        print(f"  Ranking per cluster (restricted pool): {parts}", flush=True)

                    for cid in range(scale_clusters):
                        cg = cluster_geos_map[cid]
                        if restrict_donors:
                            if len(cg) == 1:
                                if verbose:
                                    print(f"  [Warning] C{cid}: only 1 geo — no donor pool. Skipping.", flush=True)
                                cluster_ranking[cid] = []
                                continue
                            if len(cg) == 2 and verbose:
                                print(f"  [Warning] C{cid}: only 2 geos — one treatment, one control.", flush=True)
                            cluster_combos = [[g] for g in cg]
                            cluster_results = _evaluate_combinations(
                                df_rank, cg, cluster_combos,
                                all_treatment_geos=set(),
                                method=method,
                                alpha_grid=[elasticnet_alpha], l1_grid=[elasticnet_l1_ratio],
                                verbose=verbose, desc=f"C{cid}", n_jobs=n_jobs,
                            )
                            cluster_results.sort(key=lambda x: x["ser"])
                            cluster_ranking[cid] = [r["treatment"][0] for r in cluster_results]
                        else:
                            # Sequential mode: rank within cluster with intra-cluster donors.
                            # Phase 2 (global donors) is tried later per-cluster if Phase 1 fails.
                            cluster_combos = [[g] for g in cg]
                            cluster_results = _evaluate_combinations(
                                df_rank, cg, cluster_combos,
                                all_treatment_geos=set(),
                                method=method,
                                alpha_grid=[elasticnet_alpha], l1_grid=[elasticnet_l1_ratio],
                                verbose=verbose, desc=f"C{cid}", n_jobs=n_jobs,
                            )
                            cluster_results.sort(key=lambda x: x["ser"])
                            cluster_ranking[cid] = [r["treatment"][0] for r in cluster_results]

                    if verbose:
                        print(f"  Rankings built.\n")

                    # ── Sequential algorithm when restrict_donors=False ───────
                    if not restrict_donors:
                        cluster_corr_map = _cluster_mean_corr(
                            df, cluster_geos_map, date_col, start_date, end_date
                        )
                        sorted_cids = sorted(
                            range(scale_clusters),
                            key=lambda c: cluster_corr_map.get(c, 0.0),
                            reverse=True,
                        )
                        if verbose:
                            order_str = "  ".join(
                                f"C{c}({cluster_corr_map.get(c, 0):.2f})"
                                for c in sorted_cids
                            )
                            print(f"  Order (μ_corr↓): {order_str}\n", flush=True)
                        refined_clusters, fallback_names = _sequential_cluster_design(
                            sorted_cids, cluster_corr_map, cluster_ranking,
                            cluster_geos_map, geo_cluster, geos,
                            filepath, date_col, df_pre, df,
                            start_date, end_date, n_folds, experiment_days,
                            experiment_type, check_ghost_lift,
                            method, check_oof, verbose,
                            r2_threshold=r2_threshold, gap_threshold=gap_threshold,
                            wape_threshold=wape_threshold, wdist_threshold=wdist_threshold,
                        )
                        n_consolidated = len(refined_clusters) - len(fallback_names)
                        if verbose:
                            fb_str = (
                                f" + {len(fallback_names)} fallback"
                                f" ({', '.join(fallback_names)})"
                                if fallback_names else ""
                            )
                            print(
                                f"\n  Result: {n_consolidated} consolidated"
                                f"{fb_str} → {len(refined_clusters)} total slots."
                            )
                        # Populate locked_clusters so the DESIGN QUALITY section
                        # can compute n_consolidated = len(locked_clusters) correctly.
                        fallback_geo_names = {fn.split(":")[-1] for fn in fallback_names}
                        locked_clusters = [
                            r for r in refined_clusters
                            if r["best"]["treatment"][0] not in fallback_geo_names
                        ]
                    else:
                        current_candidates = [
                            cluster_ranking[cid][0]
                            for cid in range(scale_clusters)
                            if cluster_ranking[cid]
                        ]
                        n_treat = len(current_candidates)
                        cluster_next_idx = {cid: 1 for cid in range(scale_clusters)}
                        locked_cluster_ids = set()
                        next_rank_idx = None  # unused in scale-cluster mode
                else:
                    # Initialize: top n_treat candidates from the SER ranking
                    current_candidates = list(global_ranking[:n_treat])
                    next_rank_idx = n_treat
                    cluster_next_idx = None
                    locked_cluster_ids = None

                if geo_cluster is not None and not restrict_donors:
                    # Sequential algorithm already ran — make the greedy loop a no-op.
                    max_iterations = 0
                    current_candidates = []
                    cluster_next_idx = {}
                    locked_cluster_ids = set()
                elif geo_cluster is not None:
                    max_iterations = max((len(v) for v in cluster_ranking.values()), default=1)
                else:
                    max_iterations = len(global_ranking)
                # Sequential path populates locked_clusters before this point;
                # greedy path initializes them here.
                if not (geo_cluster is not None and not restrict_donors):
                    locked_clusters = []
                locked_treatments = set()
                locked_donors = set()
                all_failed = []

                # Sequential algorithm sets refined_clusters directly; skip greedy loop.
                found = geo_cluster is not None and not restrict_donors
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
                    
                    # In scale_clusters mode always print individually (bounded iterations,
                    # cluster identity matters). Compact buffering only for plain ranking mode.
                    is_single = len(candidates_to_test) == 1 and geo_cluster is None

                    if verbose and not is_single:
                        _flush_single_buffer(single_candidate_buffer, single_candidate_skips, len(locked_treatments))
                        single_candidate_buffer.clear()
                        single_candidate_skips.clear()
                        locked_str = f"  {len(locked_treatments)} locked" if locked_treatments else ""
                        print(f"  ── Iter {iteration}{locked_str} {'─' * max(0, 48 - len(str(iteration)) - len(locked_str))}", flush=True)
                    
                    iter_results = []
                    
                    for candidate in candidates_to_test:
                        # Exclude all other treatments (locked + other new) from donor pool.
                        # In scale_clusters mode Phase 1 always restricts to cluster donors —
                        # the unrestricted retry happens in Phase 2 (restrict_donors=False only).
                        other_treatments = set(current_candidates) - {candidate}
                        if geo_cluster is not None:
                            cid = geo_cluster[candidate]
                            available_geos = [
                                g for g in cluster_geos_map[cid]
                                if g not in other_treatments
                            ]
                        else:
                            available_geos = [g for g in geos if g not in other_treatments]
                        
                        try:
                            candidate_eval = discover_geo_clusters(
                                filepath=filepath, date_col=date_col, geos=available_geos,
                                fixed_treatment=[candidate], start_date=start_date, end_date=end_date,
                                method=method, check_oof=check_oof,
                                verbose=False, show_results=False, df=df
                            )
                            current_cluster = candidate_eval[0].copy()
                        except Exception:
                            iter_results.append({"candidate": candidate, "passed": False})
                            continue
                        
                        best_cluster, best_cv_row, passed, iters = _run_oof_refinement_single(
                            current_cluster, filepath, date_col, df_pre, start_date, end_date, n_folds,
                            experiment_days=experiment_days, experiment_type=experiment_type, df=df,
                            r2_threshold=r2_threshold, gap_threshold=gap_threshold,
                            wape_threshold=wape_threshold, wdist_threshold=wdist_threshold,
                        )
                        
                        ghost_lift = False
                        if passed and experiment_type != "matched_did" and check_ghost_lift:
                            ghost_lift = _check_ghost_lift_oos(best_cluster, df_pre, date_col, experiment_days)
                            if ghost_lift:
                                passed = False
                        
                        r2_t = best_cv_row["r2_test"]
                        gap_t = best_cv_row["r2_train"] - r2_t
                        
                        cid_of_candidate = geo_cluster[candidate] if geo_cluster is not None else None
                        iter_results.append({
                            "candidate": candidate, "passed": passed,
                            "raw": current_cluster, "best": best_cluster, "cv": best_cv_row,
                            "r2": r2_t, "gap": gap_t, "ghost_lift": ghost_lift,
                            "cluster_id": cid_of_candidate,
                        })

                        if verbose and not is_single:
                            status = _oof_status(
                                best_cv_row or {}, passed, ghost_lift,
                                r2_threshold, gap_threshold, wape_threshold, wdist_threshold,
                            )
                            cid_col = f"C{cid_of_candidate}" if cid_of_candidate is not None else " "
                            wape_t = (best_cv_row or {}).get("wape_test", 0.0)
                            wdst_t = (best_cv_row or {}).get("wdist_pct", 0.0)
                            print(f"    {cid_col}  {candidate:<22}  {status}  R²={r2_t:.3f}  Gap={gap_t:.3f}  WAPE={wape_t:.3f}  Wdst={wdst_t:.1f}%", flush=True)
                    
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
                            if locked_cluster_ids is not None:
                                locked_cluster_ids.add(geo_cluster[r["candidate"]])
                            if r.get("best"):
                                locked_donors.update(r["best"].get("control", []))
                        else:
                            # It caused a consolidated ghost lift, so we reject it
                            r["passed"] = False
                            r["ghost_lift"] = True
                            newly_failed.append(r)
                            if verbose and not is_single:
                                cid_r = geo_cluster[r["candidate"]] if geo_cluster is not None else ""
                                cid_col = f"C{cid_r}" if cid_r != "" else " "
                                print(f"    {cid_col}  {r['candidate']:<22}  Rejected    consolidated ghost lift", flush=True)
                    
                    all_failed.extend([r for r in newly_failed if r.get("best") is not None])
                    
                    # Find replacements + track skipped
                    iter_skipped = []
                    new_candidates = sorted(locked_treatments)

                    if len(locked_clusters) < n_treat:
                        if geo_cluster is not None:
                            # Scale-cluster mode: replace per cluster
                            for cid in range(scale_clusters):
                                if cid in locked_cluster_ids:
                                    continue  # already has a locked treatment
                                # Check if cluster still has a candidate in the queue
                                already_queued = any(
                                    geo_cluster.get(c) == cid
                                    for c in new_candidates
                                    if c not in locked_treatments
                                )
                                if already_queued:
                                    continue
                                # Pull next available geo from this cluster
                                while cluster_next_idx[cid] < len(cluster_ranking[cid]):
                                    next_geo = cluster_ranking[cid][cluster_next_idx[cid]]
                                    cluster_next_idx[cid] += 1
                                    if next_geo in locked_treatments or next_geo in locked_donors:
                                        iter_skipped.append(next_geo)
                                        continue
                                    new_candidates.append(next_geo)
                                    break
                        else:
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
                            # Individual results were printed in real-time. Print the footer.
                            if geo_cluster is not None:
                                locked_cids  = sorted(locked_cluster_ids)
                                pending_cids = [c for c in range(scale_clusters) if c not in locked_cluster_ids]
                                locked_str   = " ".join(f"C{c}" for c in locked_cids) or "—"
                                pending_str  = " ".join(f"C{c}" for c in pending_cids) or "—"
                                skip_str = f"  (skipped: {', '.join(iter_skipped)})" if iter_skipped else ""
                                print(f"    → Locked: {locked_str}  |  Pending: {pending_str}{skip_str}")
                            else:
                                n_cons = len(locked_clusters)
                                parts = [f"Consolidated {n_cons}"]
                                if iter_skipped:
                                    parts.append(f"skipped {len(iter_skipped)} donor-blocked ({', '.join(iter_skipped)})")
                                print(f"    → {' | '.join(parts)}")
                    
                    # Check if all slots are filled
                    if len(locked_clusters) >= n_treat:
                        refined_clusters = [{"raw": r.get("raw"), "best": r["best"], "cv": r["cv"],
                                             "scale_cluster_id": geo_cluster[r["candidate"]] if geo_cluster is not None else None}
                                            for r in locked_clusters[:n_treat]]
                        found = True
                        # Flush remaining buffer
                        if verbose:
                            _flush_single_buffer(single_candidate_buffer, single_candidate_skips, len(locked_treatments) - len(newly_passed))
                            single_candidate_buffer.clear()
                            single_candidate_skips.clear()
                            print(f"\n  ✓ All {n_treat} slots consolidated by iteration {iteration}.")
                        break
                    
                    # Break only when there are no more non-locked candidates to test.
                    # In scale_clusters mode some clusters may be exhausted (e.g. C4 has
                    # only 2 geos) while others still have candidates — keep iterating.
                    candidates_for_next = [c for c in new_candidates if c not in locked_treatments]
                    if not candidates_for_next:
                        # Flush remaining buffer
                        if verbose:
                            _flush_single_buffer(single_candidate_buffer, single_candidate_skips, len(locked_treatments))
                            single_candidate_buffer.clear()
                            single_candidate_skips.clear()
                            print(f"\n  ✗ All candidates exhausted after iteration {iteration}.")
                        break
                    
                    current_candidates = new_candidates
                
                if not found:
                    # ── Phase 2: unrestricted donors for failed clusters ──────────────
                    # Only when restrict_donors=False and scale_clusters is active.
                    # Treatment candidates still come from the pre-assigned cluster;
                    # donor pool is now the full geo pool (minus locked treatments).
                    if geo_cluster is not None and not restrict_donors:
                        pending_cids = [cid for cid in range(scale_clusters) if cid not in locked_cluster_ids]
                        if pending_cids and verbose:
                            cids_str = " ".join(f"C{c}" for c in pending_cids)
                            print(f"\n  [Phase 2 — Unrestricted Donors]  Retrying: {cids_str}\n")

                        for cid in pending_cids:
                            for candidate in cluster_ranking[cid]:
                                if candidate in locked_treatments:
                                    continue
                                # Full donor pool — only exclude confirmed treatments
                                avail_p2 = [g for g in geos if g not in locked_treatments and g != candidate]
                                try:
                                    ceval_p2 = discover_geo_clusters(
                                        filepath=filepath, date_col=date_col, geos=avail_p2,
                                        fixed_treatment=[candidate], start_date=start_date, end_date=end_date,
                                        method=method, check_oof=check_oof,
                                        verbose=False, show_results=False, df=df
                                    )
                                    cl_p2 = ceval_p2[0].copy()
                                except Exception:
                                    continue
                                best_p2, cv_p2, passed_p2, _ = _run_oof_refinement_single(
                                    cl_p2, filepath, date_col, df_pre, start_date, end_date,
                                    n_folds, experiment_days=experiment_days,
                                    experiment_type=experiment_type, df=df,
                                    r2_threshold=r2_threshold, gap_threshold=gap_threshold,
                                    wape_threshold=wape_threshold, wdist_threshold=wdist_threshold,
                                )
                                ghost_p2 = False
                                if passed_p2 and experiment_type != "matched_did" and check_ghost_lift:
                                    trial_p2 = [item["best"] for item in locked_clusters] + [best_p2]
                                    ghost_p2 = _check_ghost_lift_oos(trial_p2, df_pre, date_col, experiment_days)
                                    if ghost_p2:
                                        passed_p2 = False

                                r2_p2  = cv_p2["r2_test"]
                                gap_p2 = cv_p2["r2_train"] - r2_p2
                                if verbose:
                                    st = _oof_status(cv_p2, passed_p2, ghost_p2, r2_threshold, gap_threshold, wape_threshold, wdist_threshold)
                                    print(f"    C{cid}  {candidate:<22}  {st}  R²={r2_p2:.3f}  Gap={gap_p2:.3f}  WAPE={cv_p2.get('wape_test', 0.0):.3f}  Wdst={cv_p2.get('wdist_pct', 0.0):.1f}%", flush=True)

                                if passed_p2:
                                    locked_clusters.append({
                                        "candidate": candidate, "passed": True,
                                        "raw": cl_p2, "best": best_p2, "cv": cv_p2,
                                        "r2": r2_p2, "gap": gap_p2, "ghost_lift": False,
                                    })
                                    locked_treatments.add(candidate)
                                    locked_cluster_ids.add(cid)
                                    locked_donors.update(best_p2.get("control", []))
                                    if verbose:
                                        print(f"    → C{cid} resolved with unrestricted donors.")
                                    break  # move to next pending cluster

                    # Assemble: locked clusters (Phase 1 + Phase 2) + fallback
                    refined_clusters = [{"raw": r.get("raw"), "best": r["best"], "cv": r["cv"],
                                         "scale_cluster_id": geo_cluster[r["candidate"]] if geo_cluster is not None else None}
                                        for r in locked_clusters]

                    fallback_names = []
                    if geo_cluster is not None:
                        # For still-pending clusters, pick best failed geo from that cluster
                        pending_cids = [cid for cid in range(scale_clusters) if cid not in locked_cluster_ids]
                        for cid in pending_cids:
                            cluster_failed = [
                                r for r in all_failed
                                if r.get("best") is not None and geo_cluster.get(r["candidate"]) == cid
                            ]
                            if cluster_failed:
                                cluster_failed.sort(key=lambda x: x["cv"]["r2_test"], reverse=True)
                                best = cluster_failed[0]
                                refined_clusters.append({
                                    "raw": best.get("raw"), "best": best["best"], "cv": best["cv"],
                                    "scale_cluster_id": cid,
                                })
                                fallback_names.append(f"C{cid}:{best['candidate']}")
                            else:
                                if verbose:
                                    print(f"    [Warning] C{cid}: no valid candidate in Phase 1 or Phase 2 — slot left empty.")
                    else:
                        # Regular ranking mode: pick best failed overall
                        remaining = n_treat - len(refined_clusters)
                        if remaining > 0 and all_failed:
                            all_failed.sort(key=lambda x: x["cv"]["r2_test"], reverse=True)
                            for r in all_failed[:remaining]:
                                refined_clusters.append({"raw": r.get("raw"), "best": r["best"], "cv": r["cv"],
                                                         "scale_cluster_id": None})
                                fallback_names.append(r["candidate"])

                    if verbose:
                        fb_str = f" + {len(fallback_names)} fallback ({', '.join(fallback_names)})" if fallback_names else ""
                        print(f"\n  Result: {len(locked_clusters)} consolidated{fb_str} → {len(refined_clusters)} total slots.")
                
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
                                geo_name = fn.split(":")[-1]  # strip "C0:" prefix if present
                                for item in refined_clusters:
                                    if item.get("best") and item["best"]["treatment"][0] == geo_name:
                                        fb_details.append(f"{geo_name} R²={item['cv']['r2_test']:.4f}")
                            fb_str = "; ".join(fb_details) if fb_details else f"{n_fallback} cluster(s)"
                            print(f"  Fallback      : {n_fallback} cluster{'s' if n_fallback > 1 else ''}{'':10}({fb_str})")
                        print(f"  {'─' * 70}")
            
            clusters = []
            for item in refined_clusters:
                c = item["best"].copy()
                if item.get("scale_cluster_id") is not None:
                    c["scale_cluster_id"] = item["scale_cluster_id"]
                clusters.append(c)
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
                    method=method, check_oof=check_oof,
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
                        eval_cluster, filepath, date_col, df_pre, start_date, end_date, n_folds,
                        experiment_days=experiment_days, experiment_type=experiment_type, df=df,
                        r2_threshold=r2_threshold, gap_threshold=gap_threshold,
                        wape_threshold=wape_threshold, wdist_threshold=wdist_threshold,
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
            use_bootstrap_mde=use_bootstrap_mde,
            df=df
        )

        # 4d. Compact verbose per scenario
        if verbose:
            _print_scenario_table(clusters, duration, mde, cv_summary, total_geos=n_geos, experiment_days=experiment_days, experiment_type=experiment_type, df=df_pre)

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
