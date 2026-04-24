import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from reallift.utils.reporting import generate_cleaning_report

def clean_geo_data(
    data, 
    date_col: str, 
    imputation_method: str = 'interpolation', 
    constant_value: float = 1e-3, 
    verbose: bool = True,
    plot: bool = False,
    save_csv: bool = True,
    save_pdf: bool = False,
    file_name: str = 'cleaned_geo_data.csv',
    pdf_name: str = 'cleaning_report.pdf',
    max_zero_rate: float = None,
    top_n_geos: int = None,
    keep_top_quantiles: int = None,
    exclude_geos: list = None,
    quantile_bins: int = None,
    start_date: str = None,
    end_date: str = None,
    logo: str = None
) -> pd.DataFrame:
    """
    Cleans, standardizes, and validates raw geospatial time-series data for RealLift experimentation.
    
    This function acts as the primary DataOps gateway, ensuring that raw business data is 
    statistically robust for causal inference (Synthetic Control / Log-Diff transforms).
    
    Processing Pipeline:
    1. Temporal Standardization: Coerces diverse date formats into ISO standard (YYYY-MM-DD).
    2. Chronological Sorting: Reorders the timeline and detects missing days/intervals.
    3. Geo Segregation: Separates the temporal dimension from the geospatial features.
    4. Algebraic Imputation: Handles sparse vectors (zeros/NaNs) to prevent mathematical 
       instability (negative infinity) during downstream internal log-transformations.
    5. Quality Diagnostics: Generates a 'Scorecard' mapping sparsity, volume distribution, 
       and imputation impact for every geography.
    6. Strategic Filtering: Dynamically drops low-quality or low-volume geos based on 
       user-defined thresholds (Rate, N-best, or Quantiles).
    
    Args:
        data (pd.DataFrame | str): Input data source. Can be a pandas DataFrame or 
            a system path to a CSV file.
        date_col (str): The column label representing the time dimension.
        imputation_method (str): Strategy for handling sparse cells. 
            - 'interpolation': Uses linear/time-based infilling for mid-series gaps.
            - 'constant': Fills all gaps with the `constant_value`.
        constant_value (float): The micro-residual value used as a floor for 
            imputation (Default: 1e-3). Prevents log(0) errors.
        verbose (bool): If True, renders a detailed terminal report with 
            Scorecard and Quantile Analysis.
        plot (bool): If True, generates a 'Before vs After' Matplotlib visualization 
            for the top 20 geographies.
        save_csv (bool): Whether to export the result to a physical file.
        file_name (str): The target filename/path for the export (Default: 'cleaned_geo_data.csv').
        max_zero_rate (float, optional): Maximum allowed percentage (0.0 to 1.0) of 
            missing/zero values per geo. Higher rates are dropped.
        top_n_geos (int, optional): Selects only the best N geographies based 
            on a combined Quality-Volume ranking.
        keep_top_quantiles (int, optional): Retains geographies belonging to the 
            top N volume quantiles (e.g., 2 keeps Q1 and Q2).
        exclude_geos (list, optional): Hard-exclusion list of geography names 
            to permanently remove (e.g., test regions or outliers).
        quantile_bins (int, optional): Number of segments for the volume distribution 
            analysis (e.g., 4 for Quartiles).
        start_date (str, optional): Filter the dataset to start from this date (inclusive).
            Format: 'YYYY-MM-DD'. Useful to restrict analysis to a specific sub-period.
        end_date (str, optional): Filter the dataset to end at this date (inclusive).
            Format: 'YYYY-MM-DD'.
        
    Returns:
        pd.DataFrame: A refined DataFrame indexed by date, with all selected 
            geographies as positive-valued columns.
    """
    if verbose:
        print("\n" + "="*70)
        print("========== REALLIFT DATAOPS: SCORECARD & IMPUTATION ==========")
        print("="*70)
    
    # ── 0. Input Loading ──
    if isinstance(data, str):
        if verbose:
            print(f"\n  [LOAD] Reading CSV: {data}")
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("'data' must be a filepath (str) to a CSV or a pandas DataFrame.")
    
    # ── 1. Date Formatting & Sorting ──
    try:
        # infer_datetime_format is deprecated in pandas 2.0+, relying on flexible to_datetime
        # Convert to string first to prevent integers like 20240101 being parsed as nanoseconds
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format='mixed', errors='coerce' if hasattr(pd, 'to_datetime') else None)
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col].astype(str))
        
    df = df.sort_values(by=date_col).reset_index(drop=True)

    # ── 1.5 Period Filter ──
    if start_date is not None:
        sd = pd.to_datetime(start_date)
        df = df[df[date_col] >= sd]
    if end_date is not None:
        ed = pd.to_datetime(end_date)
        df = df[df[date_col] <= ed]
    if start_date is not None or end_date is not None:
        df = df.reset_index(drop=True)
        if verbose:
            _sd = df[date_col].min().strftime('%Y-%m-%d')
            _ed = df[date_col].max().strftime('%Y-%m-%d')
            print(f"  [FILTER] Period restricted to: {_sd} → {_ed} ({len(df)} rows)")

    df_raw = df.copy()  # Snapshot of raw series for plotting

    period_start = df[date_col].min().strftime('%Y-%m-%d')
    period_end   = df[date_col].max().strftime('%Y-%m-%d')
    total_days = (df[date_col].max() - df[date_col].min()).days + 1
    actual_rows = len(df)

    if verbose:
        print(f"\n  [TIME RANGE] {period_start} → {period_end}")
        print(f"  [EXPECTED DAYS] {total_days} | [ACTUAL ROWS] {actual_rows}")
        if actual_rows < total_days:
            print("  [Warning] There are missing intermediate dates in the Series.")
            
    # ── 2. Geospatial Dimension Segregation ──
    geo_cols = [c for c in df.columns if c != date_col]
    initial_geo_count = len(geo_cols)  # Capture BEFORE any filtering
    
    # ── 2.5 Hard Exception Filter ──
    if exclude_geos:
        to_drop = [c for c in exclude_geos if c in geo_cols]
        if to_drop:
            df = df.drop(columns=to_drop)
            geo_cols = [c for c in geo_cols if c not in to_drop]
            if verbose:
                print(f"  [FILTER] Excluded {len(to_drop)} geos requested by user: {to_drop[:5]}{'...' if len(to_drop)>5 else ''}")
    
    # ── 3. Capture Original State (pre-imputation snapshot) ──
    # Mark which cells are zero or NaN — these will be imputed
    imputation_mask = (df[geo_cols] == 0) | (df[geo_cols].isna())
    original_sums = df[geo_cols].fillna(0).sum()  # original totals (zeros count as 0)
    
    # ── 4. Imputation Strategy (Anti-Log Crash) ──
    if verbose:
        print(f"\n  [IMPUTATION] Method: '{imputation_method}' | Base Padding: {constant_value}")
        print("  Executing imputation map to enforce mathematical continuity...")
        
    df[geo_cols] = df[geo_cols].replace(0, np.nan)
    
    if imputation_method == 'interpolation':
        df[geo_cols] = df[geo_cols].interpolate(method='linear', limit_direction='both')
        df[geo_cols] = df[geo_cols].fillna(constant_value)
    else:
        df[geo_cols] = df[geo_cols].fillna(constant_value)
        
    if verbose:
        print("  [OK] Imputation complete. Matrices are now non-zero positive (log-ready).")
    
    # ── 5. DataOps Scorecard (with Interpolation Impact) ──
    total_len = len(df)
    scorecard_rows = []
    
    for col in geo_cols:
        mask_col = imputation_mask[col]
        n_imputed = mask_col.sum()
        
        zero_rate = ((df[col].eq(0)).sum() + mask_col.sum()) / total_len  # pre-imputation
        nan_rate = mask_col.sum() / total_len  # NaN+Zero combined = imputed cells
        
        # Original zero count and NaN count from the mask
        orig_series_filled = original_sums[col]
        
        # Interpolation impact
        sum_interpolated = df[col][mask_col].sum() if n_imputed > 0 else 0.0
        sum_original = orig_series_filled
        sum_total = df[col].sum()
        pct_interpolated = (sum_interpolated / sum_total * 100) if sum_total > 0 else 0.0
        
        scorecard_rows.append({
            "Geo": col,
            "N_Imputed": int(n_imputed),
            "Zero_Rate": nan_rate * 100,
            "Sum_Interpolated": sum_interpolated,
            "Sum_Original": sum_original,
            "Pct_Interpolated": pct_interpolated
        })
        
    score_df = pd.DataFrame(scorecard_rows).sort_values(
        by=["Zero_Rate", "Sum_Original"], 
        ascending=[True, False]
    )
        
    vol_df = None
    if quantile_bins is not None and len(score_df) >= quantile_bins:
        # Sort strictly by volume descending to bucket into quantiles
        vol_df = score_df.sort_values(by="Sum_Original", ascending=False).reset_index(drop=True)
        try:
            # Divide index into equal buckets
            vol_df['Quantile'] = pd.qcut(vol_df.index, q=quantile_bins, labels=[f"Q{i}" for i in range(1, quantile_bins+1)])
        except Exception:
            chunks = np.array_split(vol_df.index, quantile_bins)
            vol_df['Quantile'] = "Q"
            for i, chunk in enumerate(chunks):
                vol_df.loc[chunk, 'Quantile'] = f"Q{i+1}"
                
    if verbose:
        if vol_df is not None:
            print(f"\n  [QUANTILE ANALYSIS] Segmented by Original Volume ({quantile_bins} bins)")
            global_vol = vol_df["Sum_Original"].sum()
            
            q_hdr = f"  {'Quantile':<10} | {'Geos #':<8} | {'Avg % Zeros':<12} | {'Σ Volume':<18} | {'% of Total':<12} | {'Cumulative %':<14} | {'Avg Vol/Geo':<15}"
            print(q_hdr)
            print("  " + "─"*105)
            
            cum_pct = 0.0
            for i in range(1, quantile_bins+1):
                q = f"Q{i}"
                q_data = vol_df[vol_df['Quantile'] == q]
                if q_data.empty: continue
                
                n_geos_q = len(q_data)
                avg_zeros = f"{q_data['Zero_Rate'].mean():.2f}%"
                sum_vol = q_data['Sum_Original'].sum()
                sum_vol_str = f"{sum_vol:,.2f}"
                
                pct_val = (sum_vol / global_vol * 100) if global_vol > 0 else 0.0
                cum_pct += pct_val
                
                pct_total = f"{pct_val:.2f}%"
                cum_pct_str = f"{cum_pct:.2f}%"
                avg_vol_geo = f"{(sum_vol / n_geos_q):,.2f}"
                
                print(f"  {q:<10} | {n_geos_q:<8} | {avg_zeros:<12} | {sum_vol_str:<18} | {pct_total:<12} | {cum_pct_str:<14} | {avg_vol_geo:<15}")
        
    # ── 5.5 Optional Sparsity Filter ──
    if max_zero_rate is not None:
        rates = imputation_mask.sum() / len(imputation_mask)
        sparse_cols = rates[rates > max_zero_rate].index.tolist()
        if sparse_cols:
            df = df.drop(columns=sparse_cols)
            geo_cols = [c for c in geo_cols if c not in sparse_cols]
            if verbose:
                print(f"  [FILTER] 'max_zero_rate={max_zero_rate}': Dropped {len(sparse_cols)} geos exceeding threshold.")
                print(f"  [FILTER] Remaining eligible geos: {len(geo_cols)}")
                print()
                
    # ── 5.6 Optional Top N Geos Selector ──
    if top_n_geos is not None and top_n_geos < len(geo_cols):
        ordered_valid_geos = [c for c in score_df['Geo'].tolist() if c in geo_cols]
        kept_geos = ordered_valid_geos[:top_n_geos]
        dropped_topn = [c for c in geo_cols if c not in kept_geos]
        
        df = df.drop(columns=dropped_topn)
        geo_cols = kept_geos
        
        if verbose:
            print(f"  [FILTER] 'top_n_geos={top_n_geos}': Selected the {top_n_geos} best geos.")
            print(f"  [FILTER] Dropped {len(dropped_topn)} geos out of Top N Quality/Volume ranking.")
            print()
            
    # ── 5.6.5 Optional Top Quantiles Selector ──
    if keep_top_quantiles is not None and vol_df is not None:
        valid_qs = [f"Q{i}" for i in range(1, keep_top_quantiles + 1)]
        kept_geos_q = vol_df[vol_df['Quantile'].isin(valid_qs)]['Geo'].tolist()
        
        kept_geos_q = [c for c in kept_geos_q if c in geo_cols]
        dropped_topq = [c for c in geo_cols if c not in kept_geos_q]
        
        df = df.drop(columns=dropped_topq)
        geo_cols = kept_geos_q
        
        if verbose:
            print(f"  [FILTER] 'keep_top_quantiles={keep_top_quantiles}': Kept {len(geo_cols)} geos belonging to top {keep_top_quantiles} volume quantiles.")
            print(f"  [FILTER] Dropped {len(dropped_topq)} geos mapped to lower quantiles.")
            print()

    # ── 5.7 Compute Final Summary (always, for PDF and verbose) ──
    # Stats for the FULL original matrix (before geo filtering)
    all_geo_cols_in_mask = list(imputation_mask.columns)
    total_cells_original = total_len * len(all_geo_cols_in_mask)
    total_imputed_original = int(imputation_mask.sum().sum())
    global_pct_original = total_imputed_original / total_cells_original * 100 if total_cells_original > 0 else 0
    
    # Stats for the SURVIVING geos only
    surviving_mask = imputation_mask[[c for c in geo_cols if c in imputation_mask.columns]]
    total_cells = total_len * len(geo_cols)
    total_imputed = int(surviving_mask.sum().sum())
    global_pct = total_imputed / total_cells * 100 if total_cells > 0 else 0
    global_vol = score_df["Sum_Original"].sum()
    
    final_score_df = score_df[score_df['Geo'].isin(geo_cols)]
    n_sel = len(final_score_df)
    sum_vol = final_score_df["Sum_Original"].sum() if not final_score_df.empty else 0
    avg_zeros_sel = f"{final_score_df['Zero_Rate'].mean():.2f}%" if not final_score_df.empty else "0.00%"
    sum_vol_str = f"{sum_vol:,.2f}"
    pct_total_sel = f"{(sum_vol / global_vol * 100):.2f}%" if global_vol > 0 else "0.00%"
    avg_vol_sel = f"{(sum_vol / n_sel):,.2f}" if n_sel > 0 else "0.00"

    if verbose:
        print(f"\n  [SUMMARY] Full Matrix: {total_imputed_original:,} of {total_cells_original:,} cells imputed ({global_pct_original:.2f}%)")
        print(f"  [SUMMARY] Selected Geos: {total_imputed:,} of {total_cells:,} cells imputed ({global_pct:.2f}%)")
        print()
        
        if not final_score_df.empty:
            print(f"  [GEOS SELECTED] Final Dataset Composition")
            sel_hdr = f"  {'Geos #':<8} | {'Avg % Zeros':<12} | {'Σ Volume':<18} | {'% of Total':<12} | {'Avg Vol/Geo':<15}"
            print(sel_hdr)
            print("  " + "─"*74)
            print(f"  {n_sel:<8} | {avg_zeros_sel:<12} | {sum_vol_str:<18} | {pct_total_sel:<12} | {avg_vol_sel:<15}")
            print()
            
            # ── 5.8 Final Scorecard of Surviving Geos ──
            display_n = n_sel if n_sel <= 50 else 15
            label = f"ALL {n_sel}" if display_n == n_sel else f"TOP {display_n} of {n_sel}"
            
            print(f"  [GEOS SCORECARD] {label} — IMPUTATION IMPACT (sorted by % zeros asc, then sum desc)")
            hdr = f"  {'Geo':<20} | {'Imputed':<8} | {'% Zeros':<8} | {'Σ Imputed':<15} | {'Σ Original':<15} | {'% Imputed':<10}"
            print(hdr)
            print("  " + "─"*92)
            
            for _, row in final_score_df.head(display_n).iterrows():
                geo_name = str(row['Geo'])
                if len(geo_name) > 19:
                    geo_name = geo_name[:16] + "..."
                
                n_imp = f"{row['N_Imputed']}"
                z_rate = f"{row['Zero_Rate']:.2f}%"
                s_imp = f"{row['Sum_Interpolated']:,.2f}"
                s_ori = f"{row['Sum_Original']:,.2f}"
                p_imp = f"{row['Pct_Interpolated']:.2f}%"
                
                print(f"  {geo_name:<20} | {n_imp:<8} | {z_rate:<8} | {s_imp:<15} | {s_ori:<15} | {p_imp:<10}")
                
            if display_n < n_sel:
                print(f"  ... ({n_sel - display_n} more geos omitted)")
            print()
        
    # ── 6. Visual Verification (Before vs After) ──
    if plot:
        from matplotlib.ticker import FuncFormatter

        def human_format(num, pos):
            magnitude = 0
            while abs(num) >= 1000:
                magnitude += 1
                num /= 1000.0
            return '%.1f%s' % (num, ['', 'K', 'M', 'B', 'T'][magnitude])

        # Pick the top 20 geos by total volume for better visual clarity
        top_vols = df[geo_cols].sum().sort_values(ascending=False)
        cols_to_plot = top_vols.head(20).index.tolist()
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot Before (Raw)
        for c in cols_to_plot:
            axes[0].plot(df_raw[date_col], df_raw[c], alpha=0.7, label=c)
        axes[0].set_title(f"Before Imputation (Top {len(cols_to_plot)} Geos)")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].yaxis.set_major_formatter(FuncFormatter(human_format))
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=1)
        
        # Plot After (Imputed)
        for c in cols_to_plot:
            axes[1].plot(df[date_col], df[c], alpha=0.7, label=c)
        axes[1].set_title(f"After Imputation: '{imputation_method}'")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].yaxis.set_major_formatter(FuncFormatter(human_format))
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=1)
        
        plt.tight_layout()
        plt.show()
        
    if save_pdf:
        if verbose:
            print(f"  [REPORT] Generating PDF report: '{pdf_name}'...")
        
        meta_info = {
            'start': period_start, 'end': period_end, 'days': total_days,
            'method': imputation_method, 'constant': constant_value,
            'initial_geos': initial_geo_count,
            'final_geos': len(geo_cols),
            'total_cells_original': total_cells_original,
            'imputed_cells_original': total_imputed_original,
            'imputed_pct_original': global_pct_original,
            'total_cells': total_cells,
            'imputed_cells': total_imputed,
            'imputed_pct': global_pct,
            'n_sel': n_sel, 'avg_zeros_sel': avg_zeros_sel, 
            'sum_vol_sel': sum_vol_str, 'pct_total_sel': pct_total_sel, 
            'avg_vol_sel': avg_vol_sel
        }
        
        generate_cleaning_report(
            pdf_name=pdf_name,
            meta_info=meta_info,
            vol_df=vol_df,
            final_score_df=final_score_df,
            df_raw=df_raw,
            df_cleaned=df,
            date_col=date_col,
            imputation_method=imputation_method,
            logo=logo
        )

    if save_csv:
        df.to_csv(file_name, index=False)
        if verbose:
            print(f"  [EXPORT] Cleaned data successfully exported to '{file_name}'.\n")
        
    return df

