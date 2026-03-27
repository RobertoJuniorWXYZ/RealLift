import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from reallift.utils.metrics import mape, wape, compute_r2
from reallift.config.defaults import DEFAULT_TRAIN_TEST_SPLIT

def validate_geo_groups(
    filepath,
    date_col,
    splits,
    treatment_start_date=None,
    train_test_split=DEFAULT_TRAIN_TEST_SPLIT,
    plot=True,
    export_csv=False,
    output_prefix="geo_validation",
    verbose=True
) -> dict:
    """
    Validate geo groups using train/test split.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        splits (dict or list): Split results.
        treatment_start_date (str): Date when treatment starts. Data from this date onwards will be excluded.
        train_test_split (float): Train/test split ratio.
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

        if verbose:
            print("\n==============================")
            print(f"VALIDAÇÃO DO SPLIT {i}")
            print(f"Tratamento: {treatment}")
            print(f"Controle: {controls}")
            print("==============================")

        # =====================================================
        # 🔵 MODELO NÍVEL (REAL)
        # =====================================================
        y = df[treatment].mean(axis=1)
        X = df[controls].sum(axis=1)

        # =========================
        # SPLIT
        # =========================
        n = len(df)
        test_size = int(n * train_test_split)
        train_size = n - test_size

        X_train = X.iloc[:train_size].values.reshape(-1, 1)
        y_train = y.iloc[:train_size]

        X_test = X.iloc[train_size:].values.reshape(-1, 1)
        y_test = y.iloc[train_size:]

        # =========================
        # MODELO
        # =========================
        model = LinearRegression()
        model.fit(X_train, y_train)

        # =========================
        # PREDIÇÕES
        # =========================
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_full = np.concatenate([y_pred_train, y_pred_test])

        # =========================
        # MÉTRICAS
        # =========================
        def mape(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true))

        def wape(y_true, y_pred):
            return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        r2_full = r2_score(y, y_pred_full)

        mape_train = mape(y_train, y_pred_train)
        mape_test = mape(y_test, y_pred_test)
        mape_full = mape(y, y_pred_full)

        wape_train = wape(y_train, y_pred_train)
        wape_test = wape(y_test, y_pred_test)
        wape_full = wape(y, y_pred_full)

        # =========================
        # OUTPUT DF
        # =========================
        output_df = pd.DataFrame({
            "date": df[date_col],
            "y_real": y,
            "y_pred": y_pred_full,
            "residual": y - y_pred_full,
            "is_test": [0]*train_size + [1]*test_size
        })

        all_outputs.append(output_df)

        if export_csv:
            output_df.to_csv(f"{output_prefix}_{i}.csv", index=False)

        # =========================
        # PRINT
        # =========================
        if verbose:
            print("\n--- MODELO NÍVEL (FINAL) ---")
            print(f"Train size: {train_size}")
            print(f"Test size: {test_size}")

            print(f"R2 Train: {r2_train:.4f}")
            print(f"R2 Test: {r2_test:.4f}")
            print(f"R2 Full: {r2_full:.4f}")

            print(f"MAPE Train: {mape_train:.4f}")
            print(f"MAPE Test: {mape_test:.4f}")
            print(f"MAPE Full: {mape_full:.4f}")

            print(f"WAPE Train: {wape_train:.4f}")
            print(f"WAPE Test: {wape_test:.4f}")
            print(f"WAPE Full: {wape_full:.4f}")

        # =========================
        # PLOT
        # =========================
        if plot:
            plt.figure(figsize=(14,5))

            plt.plot(df[date_col], y, label="Real")
            plt.plot(df[date_col], y_pred_full, label="Predicted")

            plt.axvline(df[date_col].iloc[train_size], linestyle="--")

            plt.title(f"Geo Validation - {treatment}")
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
            "r2_train": r2_train,
            "r2_test": r2_test,
            "r2_full": r2_full,
            "mape_test": mape_test,
            "wape_test": wape_test
        })

    summary_df = pd.DataFrame(summary_results)

    return {
        "summary": summary_df,
        "outputs": all_outputs
    }