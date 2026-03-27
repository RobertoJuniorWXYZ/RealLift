import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from reallift.utils.preprocessing import log_diff_transform, scale_data
from reallift.config.defaults import DEFAULT_ALPHA, DEFAULT_L1_RATIO, DEFAULT_COEF_THRESHOLD

def find_best_geo_split(
    filepath,
    date_col,
    geos=None,
    n_treatment=3,
    alpha=DEFAULT_ALPHA,
    l1_ratio=DEFAULT_L1_RATIO,
    coef_threshold=DEFAULT_COEF_THRESHOLD,
    fixed_treatment=None,
    verbose=True
) -> dict:
    """
    Find the best geo split for treatment and control groups.

    Parameters:
        filepath (str): Path to CSV file.
        date_col (str): Date column name.
        geos (list): List of geographies.
        n_treatment (int): Number of treatment groups.
        alpha (float): ElasticNet alpha.
        l1_ratio (float): ElasticNet l1_ratio.
        coef_threshold (float): Coefficient threshold.
        fixed_treatment (list): Fixed treatment groups.
        verbose (bool): Whether to print results.

    Returns:
        dict: Best split result.
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Aggregate by date to handle duplicate entries
    # Ensures one observation per date before running the splitting algorithm
    df = df.groupby(date_col).sum(numeric_only=True).reset_index()
    df = df.sort_values(date_col).reset_index(drop=True)

    if geos is None:
        geos = [col for col in df.columns if col != date_col]

    results = []

    alpha_grid = [0.001, 0.01, 0.1]
    l1_grid = [0.2, 0.5, 0.8]

    if fixed_treatment is not None:
        treatment_combinations = [fixed_treatment]
    else:
        treatment_combinations = combinations(geos, n_treatment)

    for treatment_comb in treatment_combinations:
        treatment_comb = list(treatment_comb)
        control_pool = [g for g in geos if g not in treatment_comb]

        try:
            df_transformed = log_diff_transform(df, treatment_comb + control_pool)

            y = df_transformed[treatment_comb].mean(axis=1).values
            X = df_transformed[control_pool].values

            X_scaled, _ = scale_data(X)

            best_local = None

            for a in alpha_grid:
                for l1 in l1_grid:
                    model = ElasticNet(alpha=a, l1_ratio=l1, max_iter=10000)
                    model.fit(X_scaled, y)

                    coefs = model.coef_
                    dynamic_threshold = max(coef_threshold, np.max(np.abs(coefs)) * 0.05)

                    selected_controls = [
                        control_pool[i]
                        for i in range(len(control_pool))
                        if abs(coefs[i]) > dynamic_threshold
                    ]

                    if len(selected_controls) == 0:
                        idx_max = np.argmax(np.abs(coefs))
                        selected_controls = [control_pool[idx_max]]

                    X_selected = df_transformed[selected_controls].values
                    X_selected_scaled, _ = scale_data(X_selected)

                    model.fit(X_selected_scaled, y)
                    y_pred = model.predict(X_selected_scaled)
                    residual = y - y_pred

                    std_residual = float(np.std(residual))
                    
                    # Handle zero variance to avoid RuntimeWarnings in corrcoef
                    if np.std(y) > 0 and np.std(y_pred) > 0:
                        corr = float(np.corrcoef(y, y_pred)[0, 1])
                    else:
                        corr = 0.0

                    candidate = {
                        "treatment": treatment_comb,
                        "control": selected_controls,
                        "std_residual": std_residual,
                        "correlation": corr,
                        "n_controls": len(selected_controls),
                        "alpha": a,
                        "l1_ratio": l1
                    }

                    if best_local is None or std_residual < best_local["std_residual"]:
                        best_local = candidate

            if best_local:
                results.append(best_local)

        except Exception as e:
            if verbose:
                print(f"Error with combination {treatment_comb}: {e}")
            continue

    if len(results) == 0:
        raise ValueError("No valid combinations found.")

    best = sorted(results, key=lambda x: x["std_residual"])[0]

    if verbose:
        print("\n=== BEST COMBINATION ===")
        print(best)

    return best

def build_geo_clusters(splits):
    """
    Expande tratamentos em clusters individuais.

    Entrada:
    {
        "treatment": [...],
        "control": [...]
    }

    Saída:
    lista de clusters independentes
    """
    clusters = []

    treatments = splits["treatment"]
    controls = splits["control"]

    for t in treatments:
        clusters.append({
            "treatment": [t],
            "control": controls
        })

    return clusters