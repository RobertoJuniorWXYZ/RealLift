# GeoExperiment.run

Execute the full geo-lift experiment analysis. Returns treatment effects, confidence intervals, and significance tests.

```python
rl.run(
    treatment_start_date=None,
    treatment_end_date=None,
    doe=None,
    scenario=None,
    perform_backtesting=None,
    start_date=None,
    end_date=None,
    geos=None,
    fixed_treatment=None,
    n_treatment=1,
    mde=0.015,
    experiment_days=None,
    n_folds=5,
    conf_level=0.95,
    random_state=None,
    ignore_treatment_start=False,
    ignore_treatment_end=False,
    plot=False,
    verbose=None,
)
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `treatment_start_date` | `str` | `None` | Campaign start date (`"YYYY-MM-DD"`). Required unless using `perform_backtesting` |
| `treatment_end_date` | `str` | `None` | Campaign end date. If `None`, uses the last date in the data |
| `doe` | `DoEResult` | `None` | Output from `rl.design()`. Resolves treatment/control geos automatically |
| `scenario` | `int` | `None` | Index of the DoE scenario to use |
| `perform_backtesting` | `dict` | `None` | Run a pre-experiment backtest instead of a real analysis. Keys: `lift` (float, injected effect) and `days` (int, window carved from the end of history). Requires `doe` + `scenario` or `fixed_treatment`. |
| `fixed_treatment` | `list` | `None` | Hard-coded treatment geos, used when `doe` is not provided |
| `start_date` | `str` | `None` | Override the analysis window start for this run only |
| `end_date` | `str` | `None` | Override the analysis window end for this run only |
| `geos` | `list` | `None` | Restrict the geo pool for this run. Defaults to all geos in `self.df` |
| `n_treatment` | `int` | `1` | Number of treatment geos to select when `doe` is not provided |
| `mde` | `float` | `0.015` | Minimum detectable effect used for power reporting |
| `experiment_days` | `list` | framework default | Duration(s) to evaluate for power analysis |
| `n_folds` | `int` | `5` | TSCV folds for synthetic control validation |
| `conf_level` | `float` | `0.95` | Bootstrap confidence level |
| `random_state` | `int` | `None` | Seed for reproducible bootstrap |
| `ignore_treatment_start` | `bool` | `False` | Ignore `treatment_start_date` boundary when fitting the synthetic control |
| `ignore_treatment_end` | `bool` | `False` | Ignore `treatment_end_date` boundary when fitting the synthetic control |
| `plot` | `bool` | `False` | Show inline plots during execution |
| `verbose` | `bool` | `None` | Override instance verbosity for this call |

## Returns

[`ExperimentResult`](ExperimentResult.md) — dict subclass with lift metrics and visualization methods.

## Examples

**Backtest (validate design before campaign):**

```python
results = rl.run(
    perform_backtesting={"lift": 0.0, "days": 28},
    doe=doe,
    scenario=0,
)
```

**Real post-campaign analysis:**

```python
rl_post = RealLift.GeoExperiment("geo_data_with_campaign.csv", date_col="date")
rl_post.clean()

results = rl_post.run(
    treatment_start_date="2025-04-01",
    treatment_end_date="2025-04-28",
    doe=doe,
    scenario=0,
)
```
