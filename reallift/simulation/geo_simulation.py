import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_geolift_data(
    start_date="2022-01-01",
    end_date="2022-06-30",
    n_geos=5,
    freq="D",
    trend_slope=0.05,
    seasonality_amplitude=10,
    seasonality_period=7,
    noise_std=2,
    treatment_geos=None,
    treatment_start="2022-05-01",
    lift=0.2,
    random_seed=42,
    plot=True,
    save_csv=False,
    save_pre_only=False,
    file_name="synthetic_geolift.csv",
    pre_file_name="synthetic_geolift_pre.csv"
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
        plot (bool): Whether to plot.
        save_csv (bool): Whether to save CSV.
        save_pre_only (bool): Whether to save pre-treatment data.
        file_name (str): CSV file name.
        pre_file_name (str): Pre-treatment CSV file name.

    Returns:
        tuple: (df, df_pre, treatment_geos)
    """
    np.random.seed(random_seed)

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    t = np.arange(len(dates))

    df = pd.DataFrame({"date": dates})

    if treatment_geos is None:
        treatment_geos = np.random.choice(
            [f"geo_{i}" for i in range(n_geos)],
            size=max(1, n_geos // 3),
            replace=False
        )

    for i in range(n_geos):
        geo_name = f"geo_{i}"

        trend = trend_slope * t
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * t / seasonality_period)
        noise = np.random.normal(0, noise_std, len(t))

        series = 50 + trend + seasonality + noise

        if geo_name in treatment_geos:
            mask = dates >= pd.to_datetime(treatment_start)
            series[mask] = series[mask] * (1 + lift)

        df[geo_name] = series

    pre_mask = df["date"] < pd.to_datetime(treatment_start)
    df_pre = df.loc[pre_mask].copy()

    if plot:
        plt.figure(figsize=(14, 6))
        for col in df.columns[1:]:
            if col in treatment_geos:
                plt.plot(df["date"], df[col], label=f"{col} (treated)", linewidth=2)
            else:
                plt.plot(df["date"], df[col], linestyle="--", alpha=0.7)

        plt.axvline(pd.to_datetime(treatment_start), color="black", linestyle=":", label="treatment start")
        plt.legend()
        plt.title("Synthetic GeoLift Data")
        plt.show()

    if save_csv:
        df.to_csv(file_name, index=False)

    if save_pre_only:
        df_pre.to_csv(pre_file_name, index=False)

    return df, df_pre, treatment_geos