import numpy as np
from sklearn.metrics import r2_score

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: MAPE value.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true))

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: WAPE value.
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared score.

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: R2 value.
    """
    return r2_score(y_true, y_pred)