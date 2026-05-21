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

        if save_csv:
            self._filepath = file_name

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Design of Experiments
    # ──────────────────────────────────────────────────────────────────────

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
        use_elasticnet=False,
        check_ghost_lift=True,
        n_jobs=None,
        verbose=None,
        save_pdf=False,
        pdf_name="doe_report.pdf",
        logo=None,
    ) -> DoEResult:
        """Run Design of Experiments. Returns a :class:`DoEResult`."""
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
            use_elasticnet=use_elasticnet,
            check_ghost_lift=check_ghost_lift,
            n_jobs=n_jobs,
            verbose=verbose,
            save_pdf=save_pdf,
            pdf_name=pdf_name,
            logo=logo,
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
