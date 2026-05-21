import pandas as pd
import numpy as np


class RealLift:
    """
    Base class for RealLift. Handles data loading, date parsing, and geo detection.

    Subclass this to build experiment-type-specific orchestrators (e.g. GeoExperiment).

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to a CSV file or a pandas DataFrame.
    date_col : str
        Name of the date column. Values must be in ISO format ``YYYY-MM-DD``.
    start_date : str, optional
        Default analysis window start (YYYY-MM-DD). Stored and passed to subclass methods.
    end_date : str, optional
        Default analysis window end (YYYY-MM-DD).
    verbose : bool, default True
        Whether to print summary information on initialization.
    """

    def __init__(
        self,
        data,
        date_col: str,
        start_date: str = None,
        end_date: str = None,
        verbose: bool = True,
    ):
        self.date_col = date_col
        self.start_date = start_date
        self.end_date = end_date
        self._verbose = verbose
        self._filepath = None

        if isinstance(data, str):
            self._filepath = data
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(
                f"'data' must be a file path (str) or pd.DataFrame, got {type(data).__name__}"
            )

        if date_col not in df.columns:
            raise ValueError(
                f"Date column '{date_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            parsed = pd.to_datetime(df[date_col].astype(str), format="ISO8601", errors="coerce")
            bad_mask = parsed.isna() & df[date_col].notna()
            if bad_mask.any():
                examples = ", ".join(f'"{v}"' for v in df.loc[bad_mask, date_col].unique()[:3])
                raise ValueError(
                    f"Date column '{date_col}' contains values not in ISO format: {examples}. "
                    f"Expected YYYY-MM-DD or YYYY-MM-DD HH:MM:SS."
                )
            df[date_col] = parsed

        n_nat = df[date_col].isna().sum()
        if n_nat > 0:
            if verbose:
                print(f"  [Warning] Dropped {n_nat} rows with unparseable dates.")
            df = df.dropna(subset=[date_col])

        df = df.sort_values(date_col).reset_index(drop=True)

        self.geos = [col for col in df.columns if col != date_col]
        if len(self.geos) < 2:
            raise ValueError(
                f"Need at least 2 geo columns, found {len(self.geos)}: {self.geos}"
            )

        self._df_raw = df.copy()
        self.df = df

        if verbose:
            date_min = df[date_col].min().strftime("%Y-%m-%d")
            date_max = df[date_col].max().strftime("%Y-%m-%d")
            print(f"\n  RealLift initialized")
            print(f"  {'-' * 50}")
            print(f"  Date column : {date_col}")
            print(f"  Date range  : {date_min} -> {date_max}")
            print(f"  Rows        : {len(df):,}")
            print(f"  Geos        : {len(self.geos)}")
            if start_date or end_date:
                sd = start_date or date_min
                ed = end_date or date_max
                print(f"  Window      : {sd} -> {ed}")
            print(f"  {'-' * 50}\n")
