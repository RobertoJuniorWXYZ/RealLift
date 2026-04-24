# `reallift.geo.validation.validate_geo_clusters`

The `validate_geo_clusters` function is the auditing engine for static or temporal *Overfitting* of candidate geography groups. Its purpose is to ensure that the chosen Synthetic Control is not simply a result of a spurious perfect fit to past noise, but that it possesses genuine *Out-of-Sample* predictive capability.

## Signature

```python
def validate_geo_clusters(
    filepath: str,
    date_col: str,
    splits: list,
    treatment_start_date: str = None,
    start_date: str = None,
    end_date: str = None,
    train_test_split: float = 0.8,
    n_folds: int = 1,
    plot: bool = True,
    export_csv: bool = False,
    output_prefix: str = "geo_validation",
    cluster_idx: int = None,
    verbose: bool = True
) -> dict
```

---

## Mathematical and Algorithmic Foundation

Validation empirically solves the bias-variance problem in the pre-treatment phase.

### Time Series Cross-Validation

If `n_folds > 1`, the function applies a routine based on `TimeSeriesSplit` preserving the arrow-of-time causality. It partitions the pre-treatment matrix into $K$ sequential sliding windows.
In each *fold*:
1. Strict convex synthetic weights ($w$) are optimized (via *SCS Solver* cooperating with `cvxpy`) rooted purely in the `Train Set` data of its respective partition.
2. This optimized matrix equation is blindly applied to infer the temporal variance counterfactual in the adjacent `Test Set` space.
3. Blind prediction residuals (OOF) guide rigorous absolute metrics like *Out-Of-Fold* $R^2$ and Mean Absolute Percentage Error (MAPE and WAPE).

### Static Trend Split

If `n_folds = 1` (default), a deterministic split defined directly by the `train_test_split` threshold is applied (usually reserving the initial 80% of time for Train, the final 20% as Virgin Test).

### Penalization Metrics (OOF R2 Gap)

The algorithm is cautious and will trigger a visible *Warning* on your terminal screen (`⚠️ High R2 gap...`) if the predictive assertiveness of the blind base falls systematically more than `0.20` points compared to training. Train R-Squared should never polarize catastrophically with final tests; severe gaps indeed constitute "Noise Signatures".

## Return (*Output*)

Returns a Python super-dictionary encompassing:
- `summary`: DataFrame with indexed arrays of `r2_train`, `r2_test`, `mape_test` crossed by each round.
- `outputs`: Serialized list of DataFrames for parametric visualization with a clean temporal matrix and `residuals`.
