# `reallift.geo.duration.estimate_duration`

The `estimate_duration` function grounds the pillars of *Power Analysis* and Sample Sizing. It is responsible for quantifying control group statistical inertia and determining experiment temporal feasibility to detect a given effect (MDE).

## Signature

```python
def estimate_duration(
    filepath: str,
    date_col: str,
    treatment_geo: str | list = None,
    control_geos: list = None,
    clusters: list = None,
    mde: float = 0.01,
    alpha: float = 0.05,
    power_target: float = 0.8,
    experiment_days: int | list = [21, 60],
    start_date: str = None,
    end_date: str = None,
    cluster_idx: int = None,
    consolidated: bool = False,
    cluster_residuals: list = None,
    verbose: bool = True
) -> dict
```

## Operation Modes

The function operates in three distinct modes depending on provided parameters:

### 1. Duration Estimation (Default)
When an `mde` is provided (e.g., 0.02), the function calculates statistical power for each day in the `experiment_days` range and identifies the **minimum number of days** needed to reach `power_target` (usually 80%).

### 2. Auto-MDE (Inverse MDE)
If `mde=None`, the function inverts the logic: for each possible duration, it calculates the **Minimum Detectable Effect (MDE)** that can be guaranteed with 80% power. Useful when flight time is fixed and you want to know test sensitivity.

### 3. Multi-Cluster Mode (DoE Orchestration)
If the `clusters` parameter is provided (output from `discover_geo_clusters`), the function automatically executes:
1. An individual analysis for each treatment cluster.
2. A **Consolidated** analysis, using average residual variance from all clusters to project experiment power as a whole.

## Mathematical Foundation

1.  **log-diff Transformation**: Applies $\log(Y_t) - \log(Y_{t-1})$ to focus on relative growth and ensure stationarity.
2.  **OLS Purification**: Performs linear regression between Treatment and Controls to remove common seasonalities. Residual standard deviation from this regression ($\sigma_{reg}$) represents "pure noise" the MDE must overcome.
3.  **Power Calculation**: Uses normal distribution to find the point where injected effect ($\Delta$) becomes distinguishable from historical noise:
    $$ (1-\beta) = \Phi \left( \frac{\Delta}{\sigma_{reg} / \sqrt{n}} - Z_{1-\alpha/2} \right) $$

## Return (*Output*)

Returns a dictionary containing the executive summary and detailed power curve:

```python
{
    "summary": {
        "mde": 0.01,                 # MDE used or calculated
        "best_days": 21,             # Suggested ideal duration
        "best_power": 0.82,          # Power achieved at best_days
        "sigma": 0.045,              # Residual noise (log-diff)
        "delta_abs": 1250.50,        # Estimated daily incremental absolute impact
        "estimated_days_needed": 19, # Exact theoretical projection via T-Student
        "auto_mde": False,           # Indicates if Auto-MDE mode was used
        "consolidated": False        # Indicates if it's a consolidated view of multiple clusters
    },
    "power_curve": pd.DataFrame,     # Day-by-day Power and MDE table
    "residuals": pd.Series           # Residual time series (useful for audit)
}
```
