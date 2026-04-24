# `reallift.pipelines.geo_pipeline.run_geo_experiment`

In the RealLift library, the `run_geo_experiment` function is the primary end-to-end analytical pipeline. It centralizes all necessary steps to consolidate the result of a Geographic A/B test **after (or during) the intervention**.

## Signature

```python
def run_geo_experiment(
    filepath: str,
    date_col: str,
    treatment_start_date: str,
    treatment_end_date: str = None,
    doe: dict = None,
    scenario: int = None,
    start_date: str = None,
    end_date: str = None,
    geos: list = None,
    n_treatment: int = 1,
    fixed_treatment: list = None,
    mde: float = 0.015,
    experiment_days: int | list = [21, 60],
    n_folds: int = 5,
    random_state: int = None,
    plot: bool = True,
    verbose: bool = True
) -> dict
```

## Integration with Design of Experiments (DoE)

One of the greatest advantages of `run_geo_experiment` is the ability to directly read the object returned by the `design_of_experiments` function.

- **`doe`**: If you pass the dictionary returned by the DoE, the pipeline will inherit both the analytical modality (Synthetic Control vs. Matched DiD) and ignore the `geos`, `n_treatment`, and `fixed_treatment` parameters, using exactly the clusters validated during planning.
- **`scenario`**: Index of the chosen scenario within the `doe` dictionary (e.g., 1 for the 10% treatment scenario, 2 for 20%, etc.).

## Analysis Windows

- **`treatment_start_date`**: Exact date when the campaign started. It divides the world into "Train" and "Test".
- **`treatment_end_date`**: (New) If you want to analyze only a portion of the post-intervention period (e.g., the first 14 days of a campaign that is still running), use this parameter to close the Lift window.

## Pipeline Stages

1. **Cluster Discovery/Recovery**: Recovers from the DoE or discovers via ElasticNet the best synthetic controls.
2. **Cross-Validation (`validate_geo_clusters`)**: Attests to the historical robustness of the series.
3. **Statistical Power (`estimate_duration`)**: Validates if the final test (now with a set timeframe) has real significance over the obtained data.
4. **Causal Calculation**: Depending on the methodology inherited from the DoE, it activates `run_synthetic_control` (Convex Optimization) or `run_matched_did` (Difference-in-Differences). It calculates Absolute and Percentage Lift with confidence intervals via **Pooled Bootstrap**.
5. **Robust Placebo (`run_placebo_tests`)**: Applies exact permutation tests on controls to calculate the **MSPE Ratio** and attest that the trend break is singular and not a cyclical market effect. It maintains total methodological coherence by mirroring the calculation method from Step 4.
6. **Visual Diagnostics**: Renders time series plots, accumulated lift, and placebo distribution.

## Verbose Report

With `verbose=True`, the function prints to the terminal:

- **CROSS-VALIDATION SUMMARY**: Backtesting metrics per cluster (Train/Test R², Train/Test MAPE, and Train/Test WAPE), with dynamic column widths adjusted to the longest geo name — no truncation.
- **CLUSTER-LEVEL INCREMENTAL IMPACT**: Consolidated table of the final result, with one row per cluster. All numeric columns adapt to the data size:

| Column | Description |
|:---|:---|
| `Treatment` | Full name of the treated geo (no truncation) |
| `Observed` | Observed sum in the post-test period |
| `Synthetic` / `Matched` | Counterfactual sum in the post-test period |
| `Lift (%)` | Percentage lift |
| `Lift (abs)` | Absolute lift |
| `CI 95% (%)` | Percentage confidence interval (Bootstrap) |
| `CI 95% (abs)` | Absolute confidence interval (Bootstrap) |
| `Sig` | Statistical significance (`[Yes]` / `[No]`) |
| `Causal` | Causality via placebo (`[Yes]` / `[No]`) |
| `MDE@Nd` | Minimum Detectable Effect for N days of the experiment |

## Return (*Output*)

```python
{
    "clusters": [...],       # Clusters used
    "results": [
        {
            "cluster": {...},
            "validation": {...},
            "duration": {...},
            "synthetic": {...},  # Lift and MSPE results
            "placebo": {...}    # Robust empirical p-value
        }
    ],
    "consolidated": {...}    # Aggregated view of all clusters
}
```
