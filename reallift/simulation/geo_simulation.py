import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_geo_data(
    start_date="2022-01-01",
    end_date="2022-06-30",
    n_geos=5,
    freq="D",
    trend_slope=0.05,
    seasonality_amplitude=10,
    seasonality_period=7,
    noise_std=2,
    treatment_geos=None,
    treatment_start=None,
    lift=0.2,
    random_seed=42,
    plot=True,
    save_csv=False,
    save_pre_only=False,
    file_name="synthetic_geolift.csv",
    pre_file_name="synthetic_geolift_pre.csv",
    base_value=50.0,
    as_integer=False,
    pre_only=False
) -> tuple:
    """
    Generate synthetic GeoLift data.

    Parameters:
        start_date (str): Start date.
        end_date (str): End date.
        n_geos (int): Number of geographies.
        freq (str): Frequency.
        trend_slope (float): Trend slope.
        seasonality_amplitude (float): Seasonality amplitude.
        seasonality_period (int): Seasonality period.
        noise_std (float): Noise standard deviation.
        treatment_geos (list): List of treated geographies.
        treatment_start (str): Treatment start date.
        lift (float): Lift effect.
        random_seed (int): Random seed.
        plot (bool): Whether to plot the generated simulation data.
        save_csv (bool): Whether to save CSV files explicitly.
        save_pre_only (bool): If True, do not save post-test data, only pre-test data.
        file_name (str): Expected output CSV file name for the complete dataset.
        pre_file_name (str): Expected output CSV file name for the pre-treatment period data.
        base_value (float or list or tuple): Base generation metric (e.g. baseline sales volume). If list [min, max], randomize between range.
        as_integer (bool): If True, casts the simulated output dataframe strictly into integers.
        pre_only (bool): If True, returns and plots only the pre-treatment period data.

    Returns:
        tuple: (df, df_pre, treatment_geos)
    """
    np.random.seed(random_seed)

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    t = np.arange(len(dates))

    df = pd.DataFrame({"date": dates})

    # Deterministic Treatment Selection to preserve RNG state
    potential_treatment_geos = np.random.choice(
        [f"geo_{i}" for i in range(n_geos)],
        size=max(1, n_geos // 3),
        replace=False
    )
    if treatment_geos is None:
        treatment_geos = list(potential_treatment_geos)
    
    # --- Date Normalization for Simulation ---
    if treatment_start is None:
        if pre_only:
            # Everything is pre-test
            treatment_start = df["date"].iloc[-1] + pd.Timedelta(days=1)
        else:
            # Traditional 80/20 split
            idx = int(len(df) * 0.8)
            treatment_start = df["date"].iloc[idx]
    
    t_start = pd.to_datetime(treatment_start)
    # -----------------------------------------

    for i in range(n_geos):
        geo_name = f"geo_{i}"

        trend = trend_slope * t
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * t / seasonality_period)
        
        if isinstance(noise_std, (list, tuple)) and len(noise_std) == 2:
            current_noise = np.random.uniform(noise_std[0], noise_std[1])
        else:
            current_noise = noise_std
            
        noise = np.random.normal(0, current_noise, len(t))

        if isinstance(base_value, (list, tuple)) and len(base_value) == 2:
            current_base = np.random.uniform(base_value[0], base_value[1])
        else:
            current_base = base_value

        series = current_base + trend + seasonality + noise

        if geo_name in treatment_geos:
            if isinstance(lift, (list, tuple)) and len(lift) == 2:
                current_lift = np.random.uniform(lift[0], lift[1])
            else:
                current_lift = lift
                
            mask = dates >= t_start
            series[mask] = series[mask] * (1 + current_lift)

        if as_integer:
            series = np.round(series).astype(int)

        df[geo_name] = series

    pre_mask = df["date"] < t_start
    df_pre = df.loc[pre_mask].copy()

    if pre_only:
        # If the user specifically said pre_only, but NO data was pre-test relative to the cut,
        # we consider the whole generated window as pre-test.
        if len(df_pre) == 0:
            df_pre = df.copy()
        df = df_pre.copy()

    if plot:
        plt.figure(figsize=(14, 6))
        for i, col in enumerate(df.columns[1:]):
            if pre_only:
                # Na fase pré-teste, todas as geos são baseline mas queremos identificá-las
                plt.plot(df["date"], df[col], linestyle="--", alpha=0.7, label=col)
            elif col in treatment_geos:
                plt.plot(df["date"], df[col], label=f"{col} (treated)", linewidth=2)
            else:
                plt.plot(df["date"], df[col], linestyle="--", alpha=0.7)

        if pre_only:
            # Legenda fora do gráfico para não poluir
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small', ncol=2 if len(df.columns) > 15 else 1)
        else:
            if not pre_only:
                plt.axvline(t_start, color="black", linestyle=":", label="treatment start")
            plt.legend()
            
        plt.title("Pre-test Historical Data" if pre_only else "Synthetic GeoLift Data")
        plt.tight_layout()
        plt.show()

    if save_csv:
        if not pre_only:
            df.to_csv(file_name, index=False)
        df_pre.to_csv(pre_file_name, index=False)
    elif save_pre_only:
        df_pre.to_csv(pre_file_name, index=False)

    return df, df_pre, treatment_geos

def generate_simulated_intervention(
    filepath,
    days,
    treatment_geos,
    lift=0.05,
    date_col="date",
    trend_slope=0.05,
    seasonality_amplitude=10,
    seasonality_period=7,
    noise_std=2,
    random_seed=42,
    plot=True,
    save_csv=False,
    file_name="simulated_intervention.csv",
    as_integer=False
) -> pd.DataFrame:
    """
    Generate a simulated post-intervention period by extending an existing CSV dataset.
    Maintains continuity of trend, seasonality, and noise.

    Parameters:
        filepath (str): Path to pre-test CSV file.
        days (int): Number of days to simulate in the post-intervention period.
        treatment_geos (list): Geographies to receive the lift.
        lift (float or list): Lift amount (constant or random range [min, max]).
        date_col (str): Column name for dates.
        trend_slope (float): Trend slope (should match pre-test).
        seasonality_amplitude (float): Seasonality amplitude (should match pre-test).
        seasonality_period (int): Seasonality period (should match pre-test).
        noise_std (float): Noise standard deviation (should match pre-test).
        random_seed (int): Random seed for the new noise.
        plot (bool): Whether to plot the full combined series.
        save_csv (bool): Whether to save the full result to a CSV file.
        file_name (str): File name for the saved CSV.
        as_integer (bool): If True, rounds results to the nearest integer.

    Returns:
        pd.DataFrame: Combined DataFrame (Pre-test + Post-test simulation).
    """
    # 1. Load Pre-Test Data
    df_pre = pd.read_csv(filepath)
    df_pre[date_col] = pd.to_datetime(df_pre[date_col], format='mixed', dayfirst=True, errors='coerce')
    df_pre = df_pre.dropna(subset=[date_col])
    df_pre = df_pre.sort_values(date_col).reset_index(drop=True)

    n_pre = len(df_pre)
    last_date = df_pre[date_col].iloc[-1]
    geos = [c for c in df_pre.columns if c != date_col]

    # 2. Setup Post-Test Timeline
    # The new t starts from n_pre
    t_post = np.arange(n_pre, n_pre + days)
    new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="D")
    
    df_post = pd.DataFrame({date_col: new_dates})
    
    np.random.seed(random_seed)

    for geo_name in geos:
        # Base Simulation (extension of structure)
        trend = trend_slope * t_post
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * t_post / seasonality_period)

        if isinstance(noise_std, (list, tuple)) and len(noise_std) == 2:
            current_noise = np.random.uniform(noise_std[0], noise_std[1])
        else:
            current_noise = noise_std
        noise = np.random.normal(0, current_noise, len(t_post))

        # We must align the baseline level. 
        # For simplicity, we assume the trend/seasonality/base were part of the original.
        # But we don't know the 'base_value' of the original data.
        # So we align the simulation based on the LAST observed value but keeping the trend.
        # Actually, using the same trend/seasonality formula with incremental 't' preserves phase.
        # But we need an offset to match the absolute values of the df_pre.
        
        # Calculate what the formula predicts for the last day of pre-period
        t_last = n_pre - 1
        last_trend = trend_slope * t_last
        last_seasonality = seasonality_amplitude * np.sin(2 * np.pi * t_last / seasonality_period)
        
        offset = df_pre[geo_name].iloc[-1] - (last_trend + last_seasonality)

        series = offset + trend + seasonality + noise

        # Apply Lift
        if geo_name in treatment_geos:
            if isinstance(lift, (list, tuple)) and len(lift) == 2:
                current_lift = np.random.uniform(lift[0], lift[1])
            else:
                current_lift = lift
            
            series = series * (1 + current_lift)

        if as_integer:
            series = np.round(series).astype(int)

        df_post[geo_name] = series

    # 3. Combine
    df_full = pd.concat([df_pre, df_post], ignore_index=True)

    if plot:
        plt.figure(figsize=(14, 6))
        for col in geos:
            if col in treatment_geos:
                plt.plot(df_full[date_col], df_full[col], label=f"{col} (treated)", linewidth=2)
            else:
                plt.plot(df_full[date_col], df_full[col], linestyle="--", alpha=0.6)

        plt.axvline(last_date, color="black", linestyle=":", label="simulated intervention start")
        plt.legend()
        plt.title(f"Simulated Intervention ({days} days)")
        plt.show()

    if save_csv:
        df_full.to_csv(file_name, index=False)

    return df_full