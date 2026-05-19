import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
    pre_only=False,
    n_zeros=0
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
        n_zeros (int): Number of zero-value holes to randomly inject across the dataset.

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

    if n_zeros > 0:
        geo_cols = [c for c in df.columns if c != "date"]
        total_cells = len(df) * len(geo_cols)
        actual_zeros = min(n_zeros, total_cells)
        
        flat_indices = np.random.choice(total_cells, size=actual_zeros, replace=False)
        for idx in flat_indices:
            r = idx // len(geo_cols)
            c = geo_cols[idx % len(geo_cols)]
            # We assign np.nan instead of 0 so matplotlib breaks the line cleanly
            df.loc[df.index[r], c] = np.nan

    pre_mask = df["date"] < t_start
    df_pre = df.loc[pre_mask].copy()

    if pre_only:
        # If the user specifically said pre_only, but NO data was pre-test relative to the cut,
        # we consider the whole generated window as pre-test.
        if len(df_pre) == 0:
            df_pre = df.copy()
        df = df_pre.copy()

    if plot:

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
    # Use only the recent tail for slope estimation — avoids long historical
    # downward trends collapsing the post-test simulation to near zero.
    trend_window = max(30, N // 4)
    t_tail = t[-trend_window:]
    y_tail = y[-trend_window:]
    slope_init, intercept_init = np.polyfit(t_tail, y_tail, 1)
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

        # Cap amplitude of each component to 1 std of the series.
        # FFT on short series (< 1 year) often assigns huge amplitudes to
        # long-period artefacts that swing the projection far below zero.
        y_std = float(np.std(y))
        for comp in components:
            comp['amplitude'] = min(comp['amplitude'], y_std)

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
    filepath=None,
    treatment_geos=None,
    days=None,
    start_date=None,
    end_date=None,
    lift=0.05,
    date_col="date",
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
    verbose=False,
    df=None
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
        treatment_geos (list): Geographies to receive the lift.
        days (int, optional): Number of days to simulate. Required if start_date/end_date not given.
        start_date (str, optional): Start date of the post-test period (e.g. '2026-01-01').
        end_date (str, optional): End date of the post-test period (e.g. '2026-04-10').
            If start_date and end_date are provided, days is computed automatically (inclusive).
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
        df (pd.DataFrame, optional): Pre-loaded DataFrame. When provided, skips CSV I/O.

    Returns:
        pd.DataFrame: Combined DataFrame (Pre-test + Post-test simulation).
    """
    # 1. Load Pre-Test Data
    if df is not None:
        df_pre = df.copy()
        # Ensure dates are datetime
        if not pd.api.types.is_datetime64_any_dtype(df_pre[date_col]):
            df_pre[date_col] = pd.to_datetime(df_pre[date_col], format='mixed', dayfirst=True, errors='coerce')
            df_pre = df_pre.dropna(subset=[date_col])
    else:
        if filepath is None:
            raise ValueError("Either 'filepath' or 'df' must be provided.")
        df_pre = pd.read_csv(filepath)
        df_pre[date_col] = pd.to_datetime(df_pre[date_col], format='mixed', dayfirst=True, errors='coerce')
        df_pre = df_pre.dropna(subset=[date_col])
    df_pre = df_pre.sort_values(date_col).reset_index(drop=True)

    n_pre = len(df_pre)
    last_date = df_pre[date_col].iloc[-1]
    geos = [c for c in df_pre.columns if c != date_col]

    # ═══════════════════════════════════════════════════════════════════════
    # REAL HISTORY MODE (days < 0)
    # Uses the last abs(days) of the pre-test data as the intervention period.
    # ═══════════════════════════════════════════════════════════════════════
    if days is not None and days < 0:
        n_sim = abs(days)
        if n_sim >= len(df_pre):
            raise ValueError(f"History too short ({len(df_pre)} days) for a {n_sim}-day backtest.")
        
        # Split data
        df_real_pre = df_pre.iloc[:-n_sim].copy()
        df_real_post = df_pre.iloc[-n_sim:].copy()
        
        # Apply Lift to treatment geos in the real-post slice
        for geo in geos:
            if geo in treatment_geos:
                if isinstance(lift, (list, tuple)) and len(lift) == 2:
                    current_lift = np.random.uniform(lift[0], lift[1])
                else:
                    current_lift = lift
                df_real_post[geo] = df_real_post[geo] * (1 + current_lift)
                
                if as_integer:
                    df_real_post[geo] = np.round(df_real_post[geo]).astype(int)

        df_full = pd.concat([df_real_pre, df_real_post], ignore_index=True)
        
        if verbose:
            start_int = df_real_post[date_col].iloc[0]
            end_int = df_real_post[date_col].iloc[-1]
            print(f"\n>>> SIMULATION MODE: REAL HISTORY BACKTEST")
            print(f">>> Intervention Period: {start_int} to {end_int} ({n_sim} days)")
        
        if plot:
            import matplotlib.ticker as ticker
            def human_format(x, pos):
                if abs(x) >= 1e9: return f'{x/1e9:.1f}B'
                if abs(x) >= 1e6: return f'{x/1e6:.1f}M'
                if abs(x) >= 1e3: return f'{x/1e3:.1f}k'
                return f'{x:,.0f}'
            formatter = ticker.FuncFormatter(human_format)


            last_date_pre = df_real_pre[date_col].iloc[-1]
            for col in geos:
                if col in treatment_geos:
                    plt.plot(df_full[date_col], df_full[col], label=f"{col} (treated)", linewidth=2)
                else:
                    plt.plot(df_full[date_col], df_full[col], linestyle="--", alpha=0.6)

            plt.axvline(last_date_pre, color="black", linestyle=":", label="real history intervention start")
            if log_scale:
                plt.yscale('symlog', linthresh=1e5)
            plt.gca().yaxis.set_major_formatter(formatter)
            plt.legend(loc='upper left')
            plt.title(f"Simulated Intervention (Backtest: {n_sim} real days)")
            plt.show()

        if save_csv:
            df_full.to_csv(file_name, index=False)

        return df_full

    # ═══════════════════════════════════════════════════════════════════════
    # FORECAST MODE (days > 0)
    # ═══════════════════════════════════════════════════════════════════════
    # 2. Resolve post-test duration
    if start_date is not None and end_date is not None:
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date)
        days = (ed - sd).days + 1  # inclusive
        new_dates = pd.date_range(start=sd, end=ed, freq="D")
    elif days is not None:
        new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="D")
    else:
        raise ValueError("Provide either 'days' or both 'start_date' and 'end_date'.")

    post_cols = {date_col: new_dates}
    
    if verbose:
        start_int = new_dates[0].strftime('%Y-%m-%d')
        end_int = new_dates[-1].strftime('%Y-%m-%d')
        print(f"\n>>> SIMULATION MODE: FORECAST PROJECTION")
        print(f">>> Intervention Period: {start_int} to {end_int} ({len(new_dates)} days)")

    np.random.seed(random_seed)

    for geo_name in geos:
        y_pre_geo = df_pre[geo_name].values.astype(float)
        dates_pre  = pd.to_datetime(df_pre[date_col])

        # ── 7-Day Moving Average Baseline ─────────────────────────────────────
        # Constant baseline based on the most recent week of pre-test data.
        baseline_val = float(np.mean(y_pre_geo[-7:]))
        
        # Structural Noise: Standard deviation of residuals from a 7-day rolling mean
        # to capture the typical daily volatility around the trend.
        rolling_mean = pd.Series(y_pre_geo).rolling(window=7).mean().bfill()
        residuals    = y_pre_geo - rolling_mean.values
        recent_res   = residuals[-60:] if len(residuals) > 60 else residuals
        
        if noise_std is not None:
            if isinstance(noise_std, (list, tuple)) and len(noise_std) == 2:
                geo_noise_std = float(np.random.uniform(noise_std[0], noise_std[1]))
            else:
                geo_noise_std = float(noise_std)
        else:
            geo_noise_std = float(np.std(recent_res))

        # Generate Forecast (Flat baseline + structural noise)
        current_noise = np.random.normal(0, geo_noise_std, days) if geo_noise_std > 0 else np.zeros(days)
        series = np.full(days, baseline_val) + current_noise
        series = np.maximum(series, 0.0) # Ensure non-negative values

        # ── Smooth Transition (Interpolation) ────────────────────────────
        # Linearly blend the first few post-test days from the last
        # pre-test value into the weekday-mean forecast, avoiding the
        # abrupt discontinuity that can cause division-by-zero downstream.
        last_pre_value = float(y_pre_geo[-1])
        blend_window   = min(7, days)          # one full weekly cycle
        for i in range(blend_window):
            alpha     = (i + 1) / (blend_window + 1)
            series[i] = last_pre_value * (1.0 - alpha) + series[i] * alpha

        # Apply Lift
        if geo_name in treatment_geos:
            if isinstance(lift, (list, tuple)) and len(lift) == 2:
                current_lift = np.random.uniform(lift[0], lift[1])
            else:
                current_lift = lift
            series = series * (1 + current_lift)

        if as_integer:
            series = np.round(series).astype(int)

        post_cols[geo_name] = series


    df_post = pd.DataFrame(post_cols)

    # 3. Combine
    df_full = pd.concat([df_pre, df_post], ignore_index=True)

    if plot:
        import matplotlib.ticker as ticker
        def human_format(x, pos):
            if abs(x) >= 1e9: return f'{x/1e9:.1f}B'
            if abs(x) >= 1e6: return f'{x/1e6:.1f}M'
            if abs(x) >= 1e3: return f'{x/1e3:.1f}k'
            return f'{x:,.0f}'
        formatter = ticker.FuncFormatter(human_format)


        for col in geos:
            if col in treatment_geos:
                plt.plot(df_full[date_col], df_full[col], label=f"{col} (treated)", linewidth=2)
            else:
                plt.plot(df_full[date_col], df_full[col], linestyle="--", alpha=0.6)

        plt.axvline(last_date, color="black", linestyle=":", label="simulated intervention start")
        if log_scale:
            # linthresh=1e5 (100k) allows seeing smaller series linearly up to 100k, 
            # then goes log for the millions.
            plt.yscale('symlog', linthresh=1e5)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.legend(loc='upper left')
        plt.title(f"Simulated Intervention ({days} days)")
        plt.show()

    if save_csv:
        df_full.to_csv(file_name, index=False)

    return df_full