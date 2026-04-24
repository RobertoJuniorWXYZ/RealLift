# `reallift.pipelines.geo_pipeline.design_of_experiments`

The `design_of_experiments` (DoE) function acts as the master interface for test planning. Its use is mandatory **before** starting any campaign, as it mathematically projects cost, time, and sensitivity (MDE) assumptions.

## Signature

```python
def design_of_experiments(
    filepath: str,
    date_col: str,
    start_date: str = None,
    end_date: str = None,
    geos: list = None,
    pct_treatment: float | list = None,
    fixed_treatment: list = None,
    mde: float = None,
    experiment_days: int | list = [21, 28, 30, 35],
    n_folds: int = 5,
    search_mode: str = "ranking",
    experiment_type: str = "synthetic_control",
    use_elasticnet: bool = False,
    n_jobs: int = None,
    verbose: bool = True
) -> dict
```

## Main Parameters

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `filepath` | `str` | **Required** | Path to the CSV file with historical data. |
| `date_col` | `str` | **Required** | Name of the date column. |
| `pct_treatment` | `float` \| `list` | `[0.1, 0.2, 0.3]` | Percentage(s) of geos for treatment (e.g., `0.2` for 20%). |
| `experiment_days` | `list` | `[21, 28, 30, 35]` | Time windows for MDE calculation. |
| `use_elasticnet` | `bool` | `False` | If `True`, uses ElasticNet for donor pre-filtering (recommended for high dimensionality). |
| `n_jobs` | `int` | `None` | Number of parallel processes for initial screening. |
| `experiment_type` | `str` | `"synthetic_control"` | Model type: `"synthetic_control"` or `"matched_did"`. |


## Pipeline Architecture (Design Level)

To support test feasibility, the pipeline processes pre-intervention metadata by orchestrating three pillars:

1. **Optimal Clustering (`discover_geo_clusters`)**: Identifies the best combinations of geos for treatment and control. In `synthetic_control` mode, it uses ElasticNet convex optimization. In `matched_did` mode, it groups based solely on the average correlation metric under identical weights (1/N).
2. **Pragmatic Evaluation (`validate_geo_clusters`)**: Runs Cross-Validation (Backtesting) on rolling windows to ensure holistic stability.
3. **Requirement Calculation (`estimate_duration`)**: Projects the MDE (Minimum Detectable Effect) for different durations (e.g., 21, 30, 60 days), allowing the choice of the scenario with the best cost-benefit ratio.

## Technical Report (*Verbosity*)

The terminal displays a detailed report for each scenario (e.g., 10%, 20% treatment) including:

- **EXPERIMENTAL SCOPE**: Shows the total market coverage (Distinct Geos, Distinct Treatments and Controls / Total Geos).
- **TEST POOL**: Lists the geographic units selected to receive treatment (names displayed in full, no truncation).
- **CONTROL DESIGN (DONOR POOL & WEIGHTS)**: Displays each cluster with its respective donor geos and importance weights. Useful for detecting if the model is balanced.
- **CROSS-VALIDATION SUMMARY**: Table with backtesting metrics per cluster (Train/Test R², MAPE, and WAPE), with dynamic column widths adjusted to the longest name.
- **MDE COMPARISON**: Final table comparing all scenarios. Includes the columns: `Distinct` (total distinct geos), **`Controls`** (distinct controls per scenario), `MDE`, `R²`, `MAPE`, and `WAPE` (in percentage format).

## Return (*Output*)

The returned dictionary aggregates inferences from all tested scenarios:

```python
{
    "experiment_type": "synthetic_control",
    "scenarios": [
        {
            "pct_treatment": 0.10,
            "n_treatment": 3,
            "treatment_pool": ["geo_A", "geo_B", "geo_C"], # Aggregate list of all treatments
            "clusters": [...],      # Clusters found
            "duration": {...},      # MDE and Power curve
            "validation": pd.DataFrame # Backtesting results (R2, MAPE)
        },
        ...
    ],
    "comparison": pd.DataFrame      # Consolidated comparative table
}
```
