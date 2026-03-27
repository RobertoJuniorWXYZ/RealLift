import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler

def log_diff_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply log transformation and differencing to specified columns.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        columns (List[str]): Columns to transform.

    Returns:
        pd.DataFrame: Transformed dataframe.
    """
    transformed = np.log(df[columns]).diff().dropna()
    return transformed

def scale_data(X: np.ndarray) -> tuple:
    """
    Standardize data using StandardScaler.

    Parameters:
        X (np.ndarray): Input data.

    Returns:
        tuple: (scaled_X, scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler