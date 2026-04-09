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


def _estimate_series_params(t, y, n_components=3):
    """
    Estimate trend, multiple seasonality components, and noise parameters 
    via FFT-guided OLS regression. This solves the 'spectral leakage' 
    issue where off-bin frequencies pollute the noise estimate.
    """
    N = len(y)
    if N < 4:
        # Fallback for very short series
        slope, intercept = np.polyfit(t, y, 1) if N > 1 else (0, y[0])
        return {'trend_slope': float(slope), 'intercept': float(intercept), 
                'seasonality_components': [], 'noise_std': 0.0}

    # 1. Detrend for frequency discovery
    slope_init, intercept_init = np.polyfit(t, y, 1)
    y_dt = y - (slope_init * t + intercept_init)

    # 2. Discover peaks via FFT
    fft_vals = np.fft.rfft(y_dt)
    freqs = np.fft.rfftfreq(N, d=1.0)
    mags = np.abs(fft_vals)
    mags[0] = 0 # ignore DC
    
    found_freqs = []
    m_copy = mags.copy()
    for _ in range(n_components):
        idx = np.argmax(m_copy)
        if m_copy[idx] <= 0: break
        
        # Parabolic interpolation for sub-bin frequency refinement
        if 0 < idx < len(mags) - 1:
            y0, y1, y2 = np.log(m_copy[idx-1] + 1e-9), np.log(m_copy[idx] + 1e-9), np.log(m_copy[idx+1] + 1e-9)
            p = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
            refined_idx = idx + p
            refined_f = refined_idx / N
        else:
            refined_f = freqs[idx]
            
        if refined_f > 0:
            found_freqs.append(refined_f)
            
        # Zero neighbors
        m_copy[max(0, idx-1):min(len(m_copy), idx+2)] = 0

    # 3. Solve for exact parameters using OLS
    # Model: y = a*t + b + sum(A_i*cos(2pi*f_i*t) + B_i*sin(2pi*f_i*t))
    # We build the matrix X
    X = np.column_stack([t, np.ones(N)])
    for f in found_freqs:
        X = np.column_stack([X, np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t)])
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        
        final_slope = coeffs[0]
        final_intercept = coeffs[1]
        
        components = []
        for i, f in enumerate(found_freqs):
            A = coeffs[2 + 2*i]
            B = coeffs[3 + 2*i]
            # Convert A*cos + B*sin back to R*cos(wt + phase)
            # A*cos(wt) + B*sin(wt) = Magn * cos(wt - phase)
            # where Magn = sqrt(A^2 + B^2) and phase = atan2(B, A)
            amplitude = np.sqrt(A**2 + B**2)
            phase = -np.arctan2(B, A)
            components.append({
                'period': 1.0/f,
                'amplitude': float(amplitude),
                'phase': float(phase)
            })
            
        y_pred = X @ coeffs
        residuals = y - y_pred
        noise_std = float(np.std(residuals))
        
        return {
            'trend_slope': float(final_slope),
            'intercept': float(final_intercept),
            'seasonality_components': components,
            'noise_std': noise_std
        }
    except:
        # Fallback to init params
        return {
            'trend_slope': float(slope_init),
            'intercept': float(intercept_init),
            'seasonality_components': [],
            'noise_std': float(np.std(y_dt))
        }


def generate_simulated_intervention(
    filepath,
    days,
    treatment_geos,
    lift=0.05,
    date_col="date",
    trend_slope=None,
    seasonality_amplitude=None,
    seasonality_period=None,
    noise_std=None,
    random_seed=42,
    plot=True,
    save_csv=False,
    file_name="simulated_intervention.csv",
    as_integer=False
) -> pd.DataFrame:
    """
    Generate a simulated post-intervention period by extending an existing CSV dataset.
    Automatically infers trend, seasonality, and noise characteristics from the
    pre-test data for each geography via linear regression and FFT analysis.

    When parameters are left as None (default), each geography gets its own
    estimated parameters, producing a natural continuation of the historical series.
    Explicit values override auto-inference for all geographies (backward compatible).

    Parameters:
        filepath (str): Path to pre-test CSV file.
        days (int): Number of days to simulate in the post-intervention period.
        treatment_geos (list): Geographies to receive the lift.
        lift (float or list): Lift amount (constant or random range [min, max]).
        date_col (str): Column name for dates.
        trend_slope (float or None): Trend slope. If None, auto-inferred per geography.
        seasonality_amplitude (float or None): Seasonality amplitude. If None, auto-inferred via FFT.
        seasonality_period (float or None): Seasonality period in days. If None, auto-inferred via FFT.
        noise_std (float or None): Noise standard deviation. If None, auto-inferred from residuals.
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
    t_pre = np.arange(n_pre)

    for geo_name in geos:
        y_pre = df_pre[geo_name].values.astype(float)

        # --- Parameter Estimation (auto-infer from pre-test data) ---
        params = _estimate_series_params(t_pre, y_pre)
        
        # Determine components to use
        # If manual overrides exist, we use a single component logic
        if seasonality_period is not None or seasonality_amplitude is not None:
            geo_period = params['seasonality_components'][0]['period'] if seasonality_period is None else seasonality_period
            geo_amplitude = params['seasonality_components'][0]['amplitude'] if seasonality_amplitude is None else seasonality_amplitude
            geo_phase = -np.pi / 2 if (seasonality_period is not None or seasonality_amplitude is not None) else params['seasonality_components'][0]['phase']
            
            use_components = [{
                'period': geo_period,
                'amplitude': geo_amplitude,
                'phase': geo_phase
            }]
        else:
            use_components = params['seasonality_components']

        # Intercept and Trend
        geo_slope = params['trend_slope'] if trend_slope is None else trend_slope
        if trend_slope is None:
            geo_intercept = params['intercept']
        else:
            # Re-calculate intercept based on manual slope to preserve the end-point continuity as best as possible
            t_last = n_pre - 1
            last_trend_val = geo_slope * t_last
            last_seas_val = sum(c['amplitude'] * np.cos(2 * np.pi * t_last / c['period'] + c['phase']) for c in use_components)
            geo_intercept = y_pre[-1] - (last_trend_val + last_seas_val)

        # Noise: use estimated or explicit
        if noise_std is not None and (isinstance(noise_std, (int, float)) and noise_std == 0):
            geo_noise_std = 0.0
        elif noise_std is None:
            geo_noise_std = params['noise_std']
        elif isinstance(noise_std, (list, tuple)) and len(noise_std) == 2:
            geo_noise_std = np.random.uniform(noise_std[0], noise_std[1])
        else:
            geo_noise_std = noise_std

        # --- Generate Post Series ---
        trend = geo_slope * t_post
        
        seasonality = np.zeros_like(t_post, dtype=float)
        for comp in use_components:
            seasonality += comp['amplitude'] * np.cos(2 * np.pi * t_post / comp['period'] + comp['phase'])
            
        noise = np.random.normal(0, geo_noise_std, len(t_post)) if geo_noise_std > 0 else np.zeros_like(t_post)

        # Smooth transition: correct for model-data gap at the boundary
        # Using the actual last pre-value vs what the sum of trend+seas+intercept predicts
        last_predicted = (geo_slope * (n_pre - 1) + geo_intercept +
                          sum(c['amplitude'] * np.cos(2 * np.pi * (n_pre - 1) / c['period'] + c['phase']) for c in use_components))
        correction = y_pre[-1] - last_predicted

        series = geo_intercept + trend + seasonality + noise + correction

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