# GeoExperiment.design

Run the Design of Experiments pipeline. Selects optimal treatment/control clusters and projects statistical power (MDE) for each scenario.

```python
rl.design(
    pct_treatment=None,
    fixed_treatment=None,
    mde=None,
    experiment_days=None,
    geos=None,
    n_folds=5,
    search_mode="ranking",
    experiment_type="synthetic_control",
    use_elasticnet=False,
    check_ghost_lift=True,
    n_jobs=None,
    verbose=None,
    save_pdf=False,
    pdf_name="doe_report.pdf",
    logo=None,
)
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pct_treatment` | `float \| list` | `None` | Treatment pool size(s) as a fraction of total geos (e.g. `[0.05, 0.10, 0.15]`). Each value creates one scenario. |
| `fixed_treatment` | `list` | `None` | Hard-coded treatment geos — skips the discovery step |
| `mde` | `float` | `None` | Target MDE for power analysis. If `None`, MDE is derived automatically from residuals |
| `experiment_days` | `list` | framework default | Durations to evaluate (e.g. `[21, 28, 35]`) |
| `geos` | `list` | `None` | Restrict the candidate geo pool. Defaults to all geos in `self.df` |
| `n_folds` | `int` | `5` | TSCV folds for synthetic control validation |
| `search_mode` | `str` | `"ranking"` | `"ranking"`, `"exhaustive"`, or `"auto"` |
| `experiment_type` | `str` | `"synthetic_control"` | `"synthetic_control"` or `"matched_did"` |
| `use_elasticnet` | `bool` | `False` | Pre-filter control pool with ElasticNet before convex optimization |
| `check_ghost_lift` | `bool` | `True` | Run OOS backtest to flag spurious effects in the design |
| `n_jobs` | `int` | `None` | Parallel workers for cluster search. `None` = auto |
| `save_pdf` | `bool` | `False` | Export a DoE PDF report |
| `pdf_name` | `str` | `"doe_report.pdf"` | Output PDF path |
| `logo` | `str` | `None` | Path to a logo image for the PDF |
| `verbose` | `bool` | `None` | Override instance verbosity for this call |

## Returns

[`DoEResult`](DoEResult.md) — dict subclass with scenario data and visualization methods.

## Example

```python
doe = rl.design(
    pct_treatment=[0.05, 0.10, 0.15],
    experiment_days=[21, 28, 35],
    check_ghost_lift=True,
    save_pdf=True,
    pdf_name="DoEReport.pdf",
)
```
