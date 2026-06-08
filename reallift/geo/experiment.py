import numpy as np
import pandas as pd

from ..base import RealLift
from .results import DoEResult, ExperimentResult


class GeoExperiment(RealLift):
    """
    Geographic incrementality experiment orchestrator.

    Inherits data loading and date parsing from :class:`RealLift` and adds
    the full geo-experiment pipeline: cleaning, Design of Experiments, causal
    inference analysis, and simulation.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to a CSV file or a pandas DataFrame.
    date_col : str
        Name of the date column.
    start_date : str, optional
        Default analysis window start (YYYY-MM-DD).
    end_date : str, optional
        Default analysis window end (YYYY-MM-DD).
    verbose : bool, default True
        Whether to print summary information on initialization.

    Examples
    --------
    >>> from reallift import GeoExperiment
    >>> rl = GeoExperiment("data.csv", date_col="date", start_date="2025-01-01")
    >>> rl.clean()
    >>> doe = rl.design(pct_treatment=[0.05, 0.10], experiment_days=[21, 28])
    >>> results = rl.run(perform_backtesting={"lift": 0.0, "days": 28}, doe=doe, scenario=0)
    """

    # ──────────────────────────────────────────────────────────────────────
    # Data
    # ──────────────────────────────────────────────────────────────────────

    def clean(
        self,
        imputation_method: str = "constant",
        constant_value: float = 1e-3,
        verbose: bool = None,
        plot: bool = False,
        save_csv: bool = True,
        save_pdf: bool = False,
        file_name: str = "cleaned_geo_data.csv",
        pdf_name: str = "cleaning_report.pdf",
        max_zero_rate: float = None,
        top_n_geos: int = None,
        keep_top_quantiles: int = None,
        exclude_geos: list = None,
        quantile_bins: int = None,
        logo: str = None,
    ) -> pd.DataFrame:
        """Clean and validate the geo data. Updates ``self.df`` in place."""
        from ..utils.data_cleaning import clean_geo_data

        if verbose is None:
            verbose = self._verbose

        fresh_df = self._df_raw.copy()

        result = clean_geo_data(
            data=fresh_df,
            date_col=self.date_col,
            imputation_method=imputation_method,
            constant_value=constant_value,
            verbose=verbose,
            plot=plot,
            save_csv=save_csv,
            save_pdf=save_pdf,
            file_name=file_name,
            pdf_name=pdf_name,
            max_zero_rate=max_zero_rate,
            top_n_geos=top_n_geos,
            keep_top_quantiles=keep_top_quantiles,
            exclude_geos=exclude_geos,
            quantile_bins=quantile_bins,
            start_date=self.start_date,
            end_date=self.end_date,
            logo=logo,
        )

        self.df = result
        self.geos = [c for c in result.columns if c != self.date_col]
        self._df_post_clean = result.copy()   # baseline for remove_outliers()

        if save_csv:
            self._filepath = file_name

        return result

    def remove_outliers(
        self,
        alpha: float = 1.5,
        plot: bool = True,
        verbose: bool = None,
    ) -> pd.DataFrame:
        """
        Remove geo-level outliers before pre-clustering.

        Detects outliers using the Tukey fence on ``log(geo_mean)`` and drops
        them from ``self.df`` and ``self.geos``.  Always resets from the
        post-clean baseline so calling this twice with different ``alpha``
        values starts fresh each time.

        Parameters
        ----------
        alpha : float, default 1.5
            IQR multiplier for the fence: ``[Q1 − alpha·IQR, Q3 + alpha·IQR]``.
            Standard Tukey fence is 1.5; use 3.0 for extreme outliers only.
        plot : bool, default True
            Render a strip chart in log space showing kept vs removed geos,
            Q1/Q3 markers, and fence boundaries.
        verbose : bool, optional
            Print outlier list and summary.  Defaults to ``self._verbose``.

        Returns
        -------
        pd.DataFrame
            Updated dataframe with outlier geo columns removed.

        Examples
        --------
        >>> rl.clean()
        >>> rl.remove_outliers(alpha=1.5)
        >>> k = rl.pre_clustering()
        >>> doe = rl.design(scale_clusters=k)
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        if verbose is None:
            verbose = self._verbose

        base_df = getattr(self, "_df_post_clean", self.df).copy()
        all_geos = [c for c in base_df.columns if c != self.date_col]

        geo_means = {g: base_df[g].mean() for g in all_geos}
        log_vals  = {g: np.log(max(geo_means[g], 1e-10)) for g in all_geos}
        lv_arr    = np.array(list(log_vals.values()))

        q1, q3 = np.percentile(lv_arr, [25, 75])
        iqr    = q3 - q1
        lo     = q1 - alpha * iqr
        hi     = q3 + alpha * iqr

        kept     = [g for g in all_geos if lo <= log_vals[g] <= hi]
        outliers = [g for g in all_geos if log_vals[g] < lo or log_vals[g] > hi]

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  OUTLIER REMOVAL  (alpha = {alpha})")
            print(f"{'=' * 60}")
            print(f"  Fence (log space) : [{lo:.3f}, {hi:.3f}]")
            print(f"  Q1 / Q3           : {q1:.3f} / {q3:.3f}   IQR = {iqr:.3f}")
            print(f"  Geos kept         : {len(kept)}/{len(all_geos)}")
            if outliers:
                rows = [(g, f"{geo_means[g]:,.0f}", f"{log_vals[g]:.3f}") for g in outliers]
                max_name = max(len(r[0]) for r in rows)
                print(f"\n  {'Geo':<{max_name}}  {'Mean':>12}  {'log(mean)':>10}")
                print(f"  {'-'*max_name}  {'─'*12}  {'─'*10}")
                for geo, mean_s, lv_s in rows:
                    side = "▲ upper" if log_vals[geo] > hi else "▼ lower"
                    print(f"  {geo:<{max_name}}  {mean_s:>12}  {lv_s:>10}  {side}")
            else:
                print("  No outliers detected.")
            print()

        if plot:
            PALETTE = {"kept": "#3B82F6", "outlier": "#EF4444"}
            with plt.style.context("dark_background"):
                fig, ax = plt.subplots(figsize=(12, 3.5))
                fig.patch.set_facecolor("black")
                ax.set_facecolor("black")

                rng = np.random.default_rng(0)
                for g in all_geos:
                    lv  = log_vals[g]
                    jit = rng.uniform(-0.18, 0.18)
                    is_out = g in outliers
                    color  = PALETTE["outlier"] if is_out else PALETTE["kept"]
                    ax.scatter(lv, jit, color=color, s=55, zorder=3,
                               alpha=0.85, linewidths=0)
                    if is_out:
                        ax.annotate(g, (lv, jit), textcoords="offset points",
                                    xytext=(0, 8), ha="center", fontsize=7,
                                    color="#EF4444", fontweight="bold")

                from matplotlib.transforms import blended_transform_factory
                btrans = blended_transform_factory(ax.transData, ax.transAxes)
                for x, label, ls in [
                    (q1, "Q1", "--"), (q3, "Q3", "--"),
                    (lo, f"Q1−{alpha}×IQR", ":"), (hi, f"Q3+{alpha}×IQR", ":"),
                ]:
                    ax.axvline(x, color="#94A3B8", linewidth=1, linestyle=ls, alpha=0.7)
                    ax.text(x, 0.01, label, ha="center", va="bottom",
                            fontsize=8, color="#94A3B8", transform=btrans)

                from matplotlib.patches import Patch
                legend = [Patch(color=PALETTE["kept"],    label=f"Kept ({len(kept)})"),
                          Patch(color=PALETTE["outlier"], label=f"Removed ({len(outliers)})")]
                ax.legend(handles=legend, loc="upper right", framealpha=0.2,
                          labelcolor="white", fontsize=9)

                ax.set_xlabel("log(geo mean)", color="#CBD5E1", fontsize=10)
                ax.set_yticks([])
                ax.set_title(f"Outlier Removal — Tukey Fence  (α = {alpha})",
                             color="white", fontweight="bold", fontsize=12, pad=14)
                ax.tick_params(colors="#CBD5E1")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                plt.tight_layout()
                plt.show()

        result = base_df.drop(columns=outliers)
        self.df   = result
        self.geos = [c for c in result.columns if c != self.date_col]
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Design of Experiments
    # ──────────────────────────────────────────────────────────────────────

    def pre_clustering(
        self,
        scale_clusters=None,
        geos=None,
        max_k=None,
        plot=True,
        verbose=True,
    ) -> int:
        """
        K-Means pre-clustering of geos by scale before :meth:`design`.

        Runs K-Means in log space on geo pre-period means and produces standard
        K-Means diagnostics: Silhouette + WCSS (elbow) selection plots, cluster
        summary table, and a strip chart of geo means coloured by cluster.

        When ``scale_clusters=None``, k is chosen automatically via Silhouette
        score subject to a minimum-k constraint derived from the log range of
        geo means.  Pass an explicit integer to inspect a fixed k.

        Run :meth:`remove_outliers` before this to exclude extreme geos from
        the candidate pool.

        Parameters
        ----------
        scale_clusters : int or None
            Explicit k to use.  ``None`` triggers automatic selection.
        geos : list, optional
            Subset of geos. Defaults to all geos in ``self.df``.
        max_k : int or None
            Upper bound for k search.  Defaults to ``min(8, n_geos // 3)``.
        plot : bool
            Render Silhouette/WCSS and strip charts.
        verbose : bool
            Print cluster summary tables.

        Returns
        -------
        int
            Suggested (or confirmed) number of scale clusters.

        Examples
        --------
        >>> k = rl.pre_clustering()
        >>> doe = rl.design(scale_clusters=k)
        """
        from ._design import compute_scale_clusters as _pre_clustering

        result = _pre_clustering(
            scale_clusters=scale_clusters,
            geos=geos or self.geos,
            start_date=self.start_date,
            end_date=self.end_date,
            date_col=self.date_col,
            max_k=max_k,
            plot=plot,
            verbose=verbose,
            df=self.df,
        )
        self._pre_clustering_result = result
        return result["suggested_k"]

    def cluster_correlations(self) -> "pd.DataFrame":
        """
        Mean intra-cluster Pearson correlation for each scale cluster.

        Computes the average of all pairwise off-diagonal correlations between
        geo time series within each cluster, over the pre-period window.
        Useful for assessing cluster cohesion before :meth:`design`.

        Returns
        -------
        pd.DataFrame
            Columns: ``cluster``, ``n_geos``, ``mean_corr``, ``median_corr``,
            ``min_corr``, ``max_corr``.

        Raises
        ------
        RuntimeError
            If :meth:`pre_clustering` has not been called yet.

        Examples
        --------
        >>> k = rl.pre_clustering(scale_clusters=5)
        >>> rl.cluster_correlations()
        """
        import numpy as np
        import pandas as pd

        if not hasattr(self, "_pre_clustering_result"):
            raise RuntimeError(
                "Call pre_clustering() before cluster_correlations()."
            )

        geo_df = self._pre_clustering_result["geo_df"]

        mask = pd.Series([True] * len(self.df), index=self.df.index)
        if self.start_date is not None:
            mask &= self.df[self.date_col] >= pd.to_datetime(self.start_date)
        if self.end_date is not None:
            mask &= self.df[self.date_col] <= pd.to_datetime(self.end_date)
        df_period = self.df.loc[mask]

        cluster_ids = sorted(
            [c for c in geo_df["cluster"].unique() if c != "outlier"]
        )

        rows = []
        for cid in cluster_ids:
            geos_in_cluster = geo_df.loc[geo_df["cluster"] == cid, "geo"].tolist()
            n = len(geos_in_cluster)
            if n < 2:
                rows.append({
                    "cluster": cid, "n_geos": n,
                    "mean_corr": np.nan, "median_corr": np.nan,
                    "min_corr": np.nan, "max_corr": np.nan,
                })
                continue
            corr = df_period[geos_in_cluster].corr().values
            off_diag = corr[~np.eye(n, dtype=bool)]
            rows.append({
                "cluster":     cid,
                "n_geos":      n,
                "mean_corr":   round(float(np.mean(off_diag)), 4),
                "median_corr": round(float(np.median(off_diag)), 4),
                "min_corr":    round(float(np.min(off_diag)), 4),
                "max_corr":    round(float(np.max(off_diag)), 4),
            })

        return pd.DataFrame(rows)

    def design(
        self,
        geos=None,
        pct_treatment=None,
        fixed_treatment=None,
        mde=None,
        experiment_days=None,
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
        verbose=None,
        save_pdf=False,
        pdf_name="doe_report.pdf",
        logo=None,
        use_bootstrap_mde=True,
        scale_clusters=None,
        restrict_donors=False,
    ) -> DoEResult:
        """Run Design of Experiments. Returns a :class:`DoEResult`.

        Parameters
        ----------
        method : str
            Donor weighting strategy. One of:

            ``"penalized_scm"`` (default)
                Single CVXPY step with L1 + L2 penalties and ``w >= 0``
                (no ``sum(w) = 1`` so L1 actively promotes sparsity).
                Weights are normalized after solving.

            ``"elastic_net"``
                Two-step: ElasticNet on log-diff data selects donors
                (coef > 0), then constrained SCM estimates final weights.

            ``"scm"``
                Classic SCM — CVXPY with ``w >= 0``, ``sum(w) = 1``,
                no regularization. All donors in the pool compete.
        """
        from ._design import design_of_experiments
        from ..config.defaults import DEFAULT_EXPERIMENT_DAYS

        if verbose is None:
            verbose = self._verbose
        if experiment_days is None:
            experiment_days = DEFAULT_EXPERIMENT_DAYS

        raw = design_of_experiments(
            filepath=self._filepath,
            date_col=self.date_col,
            start_date=self.start_date,
            end_date=self.end_date,
            geos=geos,
            pct_treatment=pct_treatment,
            fixed_treatment=fixed_treatment,
            mde=mde,
            experiment_days=experiment_days,
            n_folds=n_folds,
            search_mode=search_mode,
            experiment_type=experiment_type,
            method=method,
            elasticnet_alpha=elasticnet_alpha,
            elasticnet_l1_ratio=elasticnet_l1_ratio,
            check_ghost_lift=check_ghost_lift,
            check_oof=check_oof,
            r2_threshold=r2_threshold,
            gap_threshold=gap_threshold,
            wape_threshold=wape_threshold,
            wdist_threshold=wdist_threshold,
            n_jobs=n_jobs,
            verbose=verbose,
            save_pdf=save_pdf,
            pdf_name=pdf_name,
            logo=logo,
            use_bootstrap_mde=use_bootstrap_mde,
            scale_clusters=scale_clusters,
            restrict_donors=restrict_donors,
            df=self.df,
        )

        result = DoEResult(raw)
        result._df = self.df
        result._date_col = self.date_col
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Experiment
    # ──────────────────────────────────────────────────────────────────────

    def run(
        self,
        treatment_start_date=None,
        treatment_end_date=None,
        doe=None,
        scenario=None,
        start_date=None,
        end_date=None,
        geos=None,
        n_treatment=1,
        fixed_treatment=None,
        mde=0.015,
        experiment_days=None,
        n_folds=5,
        random_state=None,
        conf_level=0.95,
        plot=False,
        verbose=None,
        ignore_treatment_start=False,
        ignore_treatment_end=False,
        perform_backtesting=None,
    ) -> ExperimentResult:
        """
        Run the full geo-lift experiment analysis. Returns an :class:`ExperimentResult`.

        Parameters
        ----------
        perform_backtesting : dict, optional
            Run a pre-experiment backtest instead of a real analysis. Keys:
            ``lift`` (float, injected effect, default ``0.0``) and
            ``days`` (int, window carved from end of history, default ``28``).
            Requires ``doe`` + ``scenario`` or ``fixed_treatment``.
        """
        from ._experiment import run_geo_experiment
        from ..config.defaults import DEFAULT_EXPERIMENT_DAYS

        if verbose is None:
            verbose = self._verbose
        if experiment_days is None:
            experiment_days = DEFAULT_EXPERIMENT_DAYS

        run_df = self.df

        if perform_backtesting is not None:
            from ._simulation import generate_simulated_intervention

            bt_lift = perform_backtesting.get("lift", 0.0)
            bt_days = perform_backtesting.get("days", 28)

            if doe is not None and scenario is not None:
                clusters = doe["scenarios"][scenario]["clusters"]
                treatment_geos = []
                for c in clusters:
                    t = c["treatment"]
                    if isinstance(t, list):
                        treatment_geos.extend(t)
                    else:
                        treatment_geos.append(t)
            elif fixed_treatment is not None:
                treatment_geos = list(fixed_treatment)
            else:
                raise ValueError(
                    "perform_backtesting requires treatment geos — provide "
                    "'doe' + 'scenario' or 'fixed_treatment'."
                )

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"  BACKTESTING MODE")
                print(f"  Injected lift : {bt_lift:+.2%}")
                print(f"  Window        : last {bt_days} days of pre-test history")
                print(f"  Treatment geos: {treatment_geos}")
                print(f"{'=' * 60}\n")

            run_df = generate_simulated_intervention(
                treatment_geos=treatment_geos,
                days=-bt_days,
                lift=bt_lift,
                date_col=self.date_col,
                plot=plot,
                verbose=verbose,
                df=self.df,
            )

            treatment_start_date = (
                run_df[self.date_col].iloc[-bt_days].strftime("%Y-%m-%d")
            )
            treatment_end_date = (
                run_df[self.date_col].iloc[-1].strftime("%Y-%m-%d")
            )

        if treatment_start_date is None:
            raise ValueError(
                "treatment_start_date is required (or use perform_backtesting "
                "to derive it automatically)."
            )

        raw_results = run_geo_experiment(
            filepath=self._filepath,
            date_col=self.date_col,
            treatment_start_date=treatment_start_date,
            treatment_end_date=treatment_end_date,
            doe=doe,
            scenario=scenario,
            start_date=start_date or self.start_date,
            end_date=end_date or self.end_date,
            geos=geos,
            n_treatment=n_treatment,
            fixed_treatment=fixed_treatment,
            mde=mde,
            experiment_days=experiment_days,
            n_folds=n_folds,
            random_state=random_state,
            conf_level=conf_level,
            plot=plot,
            verbose=verbose,
            ignore_treatment_start=ignore_treatment_start,
            ignore_treatment_end=ignore_treatment_end,
            df=run_df,
        )

        result = ExperimentResult(raw_results)
        result._df = self.df
        result._date_col = self.date_col
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Simulation
    # ──────────────────────────────────────────────────────────────────────

    def simulate(
        self,
        treatment_geos,
        days=None,
        start_date=None,
        end_date=None,
        lift=0.05,
        trend_slope=None,
        seasonality_amplitude=None,
        seasonality_period=None,
        noise_std=None,
        random_seed=42,
        plot=True,
        log_scale=False,
        save_csv=False,
        file_name="simulated_intervention.csv",
        as_integer=False,
        verbose=None,
    ) -> pd.DataFrame:
        """Generate simulated post-intervention data. Updates ``self.df`` in place."""
        from ._simulation import generate_simulated_intervention

        if verbose is None:
            verbose = self._verbose

        result = generate_simulated_intervention(
            filepath=self._filepath,
            treatment_geos=treatment_geos,
            days=days,
            start_date=start_date,
            end_date=end_date,
            lift=lift,
            date_col=self.date_col,
            trend_slope=trend_slope,
            seasonality_amplitudes=seasonality_amplitudes,
            seasonality_period=seasonality_period,
            noise_std=noise_std,
            random_seed=random_seed,
            plot=plot,
            log_scale=log_scale,
            save_csv=save_csv,
            file_name=file_name,
            as_integer=as_integer,
            verbose=verbose,
            df=self.df,
        )

        self.df = result
        self.geos = [c for c in result.columns if c != self.date_col]

        if save_csv:
            self._filepath = file_name

        return result

    @classmethod
    def generate_data(
        cls,
        start_date="2020-01-01",
        end_date="2020-03-31",
        n_geos=27,
        freq="D",
        mean_values=[100, 500],
        trend_slope=0.00,
        seasonality_amplitudes=0.10,
        seasonality_period=7,
        noise_std=0.05,
        random_seed=42,
        n_zeros=0,
        as_integer=True,
        plot=True,
        save_csv=False,
        file_name="synthetic_geo_data.csv",
    ):
        """Generate synthetic geo time-series data. Returns a DataFrame."""
        from ._simulation import generate_geo_data

        return generate_geo_data(
            start_date=start_date,
            end_date=end_date,
            n_geos=n_geos,
            freq=freq,
            mean_values=mean_values,
            trend_slope=trend_slope,
            seasonality_amplitudes=seasonality_amplitudes,
            seasonality_period=seasonality_period,
            noise_std=noise_std,
            random_seed=random_seed,
            n_zeros=n_zeros,
            as_integer=as_integer,
            plot=plot,
            save_csv=save_csv,
            file_name=file_name,
        )

    def __repr__(self):
        n_rows = len(self.df) if self.df is not None else 0
        n_geos = len(self.geos) if self.geos else 0
        return f"GeoExperiment(date_col='{self.date_col}', rows={n_rows}, geos={n_geos})"
