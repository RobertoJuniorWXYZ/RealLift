import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from reallift.utils.metrics import mape, wape, compute_r2
from reallift.config.defaults import DEFAULT_TRAIN_TEST_SPLIT

def validate_geo_groups(
    filepath,
    date_col,
    splits,
    treatment_start_date=None,
    train_test_split=DEFAULT_TRAIN_TEST_SPLIT,
    n_folds=1,
    plot=True,
    export_csv=False,
    output_prefix="geo_validation",
    cluster_idx=None,
    verbose=True
) -> dict:
    """
    Validate geo groups using train/test split or Time Series Cross-Validation.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        splits (dict or list): Split results.
        treatment_start_date (str): Date when treatment starts. Data from this date onwards will be excluded.
        train_test_split (float): Train/test split ratio (used if n_folds=1).
        n_folds (int): Number of folds for Time Series Cross-Validation. If 1, uses a single train/test split.
        plot (bool): Whether to plot.
        export_csv (bool): Whether to export CSV.
        output_prefix (str): Output prefix.
        verbose (bool): Whether to print results.

    Returns:
        dict: Validation result.
    """
    df = pd.read_csv(filepath)

    df[date_col] = pd.to_datetime(
        df[date_col],
        format='mixed',
        dayfirst=True,
        errors='coerce'
    )
    df = df.dropna(subset=[date_col])

    # Filter to pre-treatment period if start date is provided
    # Excludes the treatment start date and everything after
    if treatment_start_date is not None:
        df = df[df[date_col] < pd.to_datetime(treatment_start_date)]
    
    # Aggregate by date to handle duplicate entries (e.g., granular user-level or transactional data)
    # This ensures a clean time-series for validation and prevents "zigzag" plots
    df = df.groupby(date_col).sum(numeric_only=True).reset_index()
    df = df.sort_values(date_col).reset_index(drop=True)

    if isinstance(splits, dict):
        splits = [splits]

    all_outputs = []
    summary_results = []

    for i, split in enumerate(splits):

        treatment = split["treatment"]
        controls = split["control"]

        current_idx = i if cluster_idx is None else cluster_idx

        if verbose:
            print(f"\n=== GEO CLUSTER VALIDATION (Cluster {current_idx}) ===")
            print(f"Treatment: {treatment}")
            print(f"Control: {controls}")

        # =====================================================
        # 🔵 MODELO NÍVEL (REAL)
        # =====================================================
        y = df[treatment].mean(axis=1)
        X = df[controls].sum(axis=1)

        # =========================
        # 🟢 CROSS-VALIDATION OR STATIC TREND VALIDATION
        # =========================
        cv_results = []
        avg_cv_r2, avg_cv_mape, avg_cv_wape = None, None, None
        avg_train_r2, avg_train_mape, avg_train_wape = None, None, None
        
        y_pred_final = np.array([])
        r2_display, r2_train_display = 0.0, 0.0
        mape_display, mape_train_display = 0.0, 0.0
        wape_display, wape_train_display = 0.0, 0.0
        split_line_idx = 0
        is_test_array = []
        header_name = ""
        plot_title = ""

        if n_folds > 1:
            # =========================
            # TIME SERIES CROSS-VALIDATION (OOF)
            # =========================
            tscv = TimeSeriesSplit(n_splits=n_folds)
            y_pred_oof = np.full(len(y), np.nan)
            first_test_idx = 0
            
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
                if fold_idx == 0:
                    first_test_idx = test_idx[0]
                
                X_train_cv = X.iloc[train_idx].values.reshape(-1, 1)
                y_train_cv = y.iloc[train_idx]
                X_test_cv = X.iloc[test_idx].values.reshape(-1, 1)
                y_test_cv = y.iloc[test_idx]

                cv_model = LinearRegression()
                cv_model.fit(X_train_cv, y_train_cv)
                
                y_pred_train_cv = cv_model.predict(X_train_cv)
                y_pred_cv = cv_model.predict(X_test_cv)
                y_pred_oof[test_idx] = y_pred_cv
                
                cv_results.append({
                    "fold": fold_idx,
                    "r2_train": r2_score(y_train_cv, y_pred_train_cv),
                    "r2_test": r2_score(y_test_cv, y_pred_cv),
                    "mape_train": mape(y_train_cv, y_pred_train_cv),
                    "mape_test": mape(y_test_cv, y_pred_cv),
                    "wape_train": wape(y_train_cv, y_pred_train_cv),
                    "wape_test": wape(y_test_cv, y_pred_cv)
                })

            avg_train_r2 = float(np.mean([r["r2_train"] for r in cv_results]))
            avg_cv_r2 = float(np.mean([r["r2_test"] for r in cv_results]))
            
            avg_train_mape = float(np.mean([r["mape_train"] for r in cv_results]))
            avg_cv_mape = float(np.mean([r["mape_test"] for r in cv_results]))
            
            avg_train_wape = float(np.mean([r["wape_train"] for r in cv_results]))
            avg_cv_wape = float(np.mean([r["wape_test"] for r in cv_results]))
            
            if verbose:
                print(f"\n--- TIME SERIES CROSS-VALIDATION ({n_folds} FOLDS) ---")
                print(f"Average Train R2: {avg_train_r2:.4f} | Average OOF R2: {avg_cv_r2:.4f}")
                print(f"Average Train MAPE: {avg_train_mape:.4f} | Average OOF MAPE: {avg_cv_mape:.4f}")
                print(f"Average Train WAPE: {avg_train_wape:.4f} | Average OOF WAPE: {avg_cv_wape:.4f}")
                
                r2_gap = avg_train_r2 - avg_cv_r2
                if r2_gap > 0.2:
                    print(f"⚠️ WARNING: High R2 gap ({r2_gap:.2f}). Potential overfitting detected.")

            # Use OOF series for output and plotting
            y_pred_final = y_pred_oof
            r2_display = avg_cv_r2
            r2_train_display = avg_train_r2
            mape_display = avg_cv_mape
            mape_train_display = avg_train_mape
            wape_display = avg_cv_wape
            wape_train_display = avg_train_wape
            split_line_idx = first_test_idx
            is_test_array = [0]*first_test_idx + [1]*(len(df) - first_test_idx)
            header_name = "CROSS-VALIDATION"
            plot_title = f"Geo Validation (Out-of-Fold CV) - {treatment}"
        else:
            # =========================
            # STATIC TREND VALIDATION (SINGLE SPLIT)
            # =========================
            n = len(df)
            test_size = int(n * train_test_split)
            train_size = n - test_size

            X_train = X.iloc[:train_size].values.reshape(-1, 1)
            y_train = y.iloc[:train_size]
            X_test = X.iloc[train_size:].values.reshape(-1, 1)
            y_test = y.iloc[train_size:]

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred_final = np.concatenate([y_pred_train, y_pred_test])

            r2_train_display = float(r2_score(y_train, y_pred_train))
            r2_display = float(r2_score(y_test, y_pred_test))
            
            mape_train_display = float(mape(y_train, y_pred_train))
            mape_display = float(mape(y_test, y_pred_test))
            
            wape_train_display = float(wape(y_train, y_pred_train))
            wape_display = float(wape(y_test, y_pred_test))
            
            split_line_idx = train_size
            is_test_array = [0]*train_size + [1]*test_size
            header_name = "STATIC TREND VALIDATION"
            plot_title = f"Geo Validation (Static Trend) - {treatment}"

            if verbose:
                print(f"\n--- {header_name} ---")
                print(f"Train size: {train_size}, Test size: {test_size}")
                print(f"Train R2: {r2_train_display:.4f} | Test R2: {r2_display:.4f}")
                print(f"Train MAPE: {mape_train_display:.4f} | Test MAPE: {mape_display:.4f}")
                print(f"Train WAPE: {wape_train_display:.4f} | Test WAPE: {wape_display:.4f}")
                
                r2_gap = r2_train_display - r2_display
                if r2_gap > 0.2:
                    print(f"⚠️ WARNING: High R2 gap ({r2_gap:.2f}). Potential overfitting detected.")

        # =========================
        # OUTPUT DF
        # =========================
        output_df = pd.DataFrame({
            "date": df[date_col],
            "y_real": y,
            "y_pred": y_pred_final,
            "residual": y - y_pred_final,
            "is_test": is_test_array
        })

        all_outputs.append(output_df)

        if export_csv:
            output_df.to_csv(f"{output_prefix}_{i}.csv", index=False)

        # =========================
        # PRINT (OUT-OF-DATE)
        # =========================
        # Removing "MODELO NÍVEL" terminology as requested. 
        # Output is now handled inside the n_folds conditional.

        # =========================
        # PLOT
        # =========================
        if plot:
            plt.figure(figsize=(14,5))

            plt.plot(df[date_col], y, label="Real")
            plt.plot(df[date_col], y_pred_final, label="Predicted (OOF)" if n_folds > 1 else "Predicted")

            if split_line_idx is not None:
                plt.axvline(df[date_col].iloc[split_line_idx], linestyle="--", color='gray', alpha=0.5)

            plt.title(plot_title)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()

        # =========================
        # SUMMARY
        # =========================
        summary_results.append({
            "treatment": treatment,
            "control": controls,
            "r2_train": r2_train_display,
            "r2_test": r2_display,
            "mape_train": mape_train_display,
            "mape_test": mape_display,
            "wape_train": wape_train_display,
            "wape_test": wape_display,
            "n_folds": n_folds
        })

    summary_df = pd.DataFrame(summary_results)

    return {
        "summary": summary_df,
        "outputs": all_outputs
    }