import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from reallift.utils.metrics import mape, wape, compute_r2
from reallift.config.defaults import DEFAULT_TRAIN_TEST_SPLIT

def validate_geo_clusters(
    filepath,
    date_col,
    splits,
    treatment_start_date=None,
    start_date=None,
    end_date=None,
    train_test_split=DEFAULT_TRAIN_TEST_SPLIT,
    n_folds=1,
    plot=True,
    export_csv=False,
    output_prefix="geo_validation",
    cluster_idx=None,
    verbose=True,
    force_equal_weights=False
) -> dict:
    """
    Validate geo groups using train/test split or Time Series Cross-Validation.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        splits (dict or list): Split results.
        treatment_start_date (str): Treatment start date for CV split point.
        start_date (str): YYYY-MM-DD date when history begins (optional).
        end_date (str): YYYY-MM-DD date when history ends (optional).
        train_test_split (float): Train/test split ratio (used if n_folds=1).
        n_folds (int): Number of folds for Time Series Cross-Validation evaluating the Convex Synthetic constraint. If 1, uses static temporal split.
        plot (bool): Whether to display a validation prediction plot per cluster.
        export_csv (bool): Whether to export CSV mapping.
        output_prefix (str): Output prefix for exported CVS.
        cluster_idx (int): Optional cluster ID assigned to the output for traceability.
        verbose (bool): Whether to print logging results.

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

    # 1. Period Filtering (General Window)
    if start_date is not None:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.to_datetime(end_date)]

    # 2. Extract context for CV split (Pre-treatment only)
    if treatment_start_date is not None:
        df = df[df[date_col] < pd.to_datetime(treatment_start_date)]
    
    # Aggregate by date to handle duplicate entries (e.g., granular user-level or transactional data)
    # This ensures a clean time-series for validation and prevents "zigzag" plots
    df = df.groupby(date_col).sum(numeric_only=True).reset_index()
    df = df.sort_values(date_col).reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset is empty after date and given filters.")

    start_date_str = df[date_col].iloc[0].strftime('%Y-%m-%d')
    end_date_str = df[date_col].iloc[-1].strftime('%Y-%m-%d')

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

            print(f"\n=== EVALUATING PERIOD ===")
            print(f"Start Date: {start_date_str}")
            print(f"End Date: {end_date_str}")

        # =====================================================
        # 🔵 PREPARE VARIABLES
        # =====================================================
        y = df[treatment].mean(axis=1).values.astype(float)
        X = df[controls].values.astype(float)
        
        def solve_cv_synthetic(X_train, y_train, X_test):
            y_mean = y_train.mean()
            if y_mean == 0: y_mean = 1e-10
            X_mean = X_train.mean(axis=0)
            X_mean[X_mean == 0] = 1e-10

            y_norm = y_train / y_mean
            X_norm = X_train / X_mean

            if force_equal_weights:
                weights = np.ones(X_train.shape[1]) / X_train.shape[1]
                intercept_val = 0.0
            else:
                w = cp.Variable(X_train.shape[1])
                alpha_intercept = cp.Variable()

                obj = cp.Minimize(cp.sum_squares(y_norm - (X_norm @ w + alpha_intercept)))
                cons = [w >= 0, cp.sum(w) == 1]
                prob = cp.Problem(obj, cons)
                try:
                    prob.solve(solver=cp.SCS, verbose=False)
                    weights = np.array(w.value).flatten()
                    intercept_val = float(alpha_intercept.value)
                except:
                    weights = np.ones(X_train.shape[1]) / X_train.shape[1]
                    intercept_val = 0.0

            X_test_norm = X_test / X_mean
            y_pred_norm_test = X_test_norm @ weights + intercept_val
            y_pred_test = y_pred_norm_test * y_mean
            
            y_pred_norm_train = X_norm @ weights + intercept_val
            y_pred_train = y_pred_norm_train * y_mean
            
            return y_pred_train, y_pred_test

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
                
                X_train_cv = X[train_idx]
                y_train_cv = y[train_idx]
                X_test_cv = X[test_idx]
                y_test_cv = y[test_idx]

                y_pred_train_cv, y_pred_cv = solve_cv_synthetic(X_train_cv, y_train_cv, X_test_cv)
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

            X_train = X[:train_size]
            y_train = y[:train_size]
            X_test = X[train_size:]
            y_test = y[train_size:]

            y_pred_train, y_pred_test = solve_cv_synthetic(X_train, y_train, X_test)
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
            "start_date": start_date_str,
            "end_date": end_date_str,
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