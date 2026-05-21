import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def generate_geo_data(
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
) -> pd.DataFrame:
    """Generate synthetic geo time-series data. Returns a DataFrame with a date column and one column per geo."""
    np.random.seed(random_seed)

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    t = np.arange(len(dates))

    df = pd.DataFrame({"date": dates})

    for i in range(n_geos):
        geo_name = f"geo_{i}"

        trend = trend_slope * t

        if isinstance(mean_values, (list, tuple)) and len(mean_values) == 2:
            current_mean = np.exp(np.random.uniform(np.log(mean_values[0]), np.log(mean_values[1])))
        else:
            current_mean = mean_values

        if isinstance(seasonality_amplitudes, (list, tuple)) and len(seasonality_amplitudes) == 2:
            current_amp = np.random.uniform(seasonality_amplitudes[0], seasonality_amplitudes[1])
        else:
            current_amp = seasonality_amplitudes

        seasonality = current_amp * current_mean * np.sin(2 * np.pi * t / seasonality_period)

        if isinstance(noise_std, (list, tuple)) and len(noise_std) == 2:
            current_noise = np.random.uniform(noise_std[0], noise_std[1])
        else:
            current_noise = noise_std

        noise = np.random.normal(0, current_noise * current_mean, len(t))

        series = current_mean + trend + seasonality + noise

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
            df.loc[df.index[r], c] = np.nan

    if save_csv:
        df.to_csv(file_name, index=False)

    if plot:
        import matplotlib.dates as mdates

        palette = [
            "#06B6D4", "#10B981", "#F59E0B", "#EF4444",
            "#8B5CF6", "#3B82F6", "#EC4899", "#14B8A6",
        ]

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(14, 5))
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            geo_cols = [c for c in df.columns if c != "date"]
            for i, col in enumerate(geo_cols):
                color = palette[i % len(palette)]
                ax.plot(df["date"], df[col], color=color, linewidth=1.4,
                        alpha=0.8, label=col)

            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

            ax.grid(True, which="major", linestyle="--", alpha=0.15, color="white")
            ax.grid(True, which="minor", linestyle=":", alpha=0.07, color="white")
            ax.tick_params(colors="#CBD5E1", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#1E1E1E")

            ncol = 2 if len(geo_cols) > 8 else 1
            ax.legend(loc="upper left", framealpha=0.4, facecolor="#111111",
                      edgecolor="#333333", labelcolor="white", fontsize=8, ncol=ncol)

            fig.text(0.5, 0.98, "Synthetic Geo Data", ha="center", va="top",
                     fontsize=14, fontweight="bold", color="white")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

    return df


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
            df_pre[date_col] = pd.to_datetime(df_pre[date_col], format="%Y-%m-%d", errors="coerce")
            df_pre = df_pre.dropna(subset=[date_col])
    else:
        if filepath is None:
            raise ValueError("Either 'filepath' or 'df' must be provided.")
        df_pre = pd.read_csv(filepath)
        df_pre[date_col] = pd.to_datetime(df_pre[date_col], format="%Y-%m-%d", errors="coerce")
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