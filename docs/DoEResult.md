# DoEResult

Returned by [`GeoExperiment.design()`](GeoExperiment.design.md). Behaves as a regular `dict` and adds visualization methods.

```python
doe = rl.design(...)
```

## Methods

### plot_cluster_fits

Pre-period synthetic control fit per cluster.

```python
doe.plot_cluster_fits(scenario=0, figsize=(14, 4))
```

### plot_consolidated_fit

Aggregated treatment vs. synthetic for the full experiment window.

```python
doe.plot_consolidated_fit(scenario=0, figsize=(12, 5))
```

### plot_power_analysis

Power curves as a function of true effect size, one curve per duration.

```python
doe.plot_power_analysis(
    scenario=0,
    durations=None,       # defaults to experiment_days from design()
    max_mde=0.05,
    power_target=0.80,
    alpha=0.05,
    figsize=(12, 7),
)
```

### plot_scenario_comparison

Side-by-side comparison table of all evaluated scenarios.

```python
doe.plot_scenario_comparison(figsize=(12, 6))
```

### plot_donor_weights

Bar chart of synthetic control donor weights per cluster.

```python
doe.plot_donor_weights(scenario=0, top_n=15, figsize=(10, 8))
```

### plot_validation_quality

TSCV R² quality scores per fold per cluster.

```python
doe.plot_validation_quality(scenario=0, figsize=(8, 8))
```

### plot_duration_mde_tradeoff

MDE vs. experiment duration tradeoff curve.

```python
doe.plot_duration_mde_tradeoff(
    durations=None,
    power_target=0.80,
    alpha=0.05,
    figsize=(12, 7),
)
```

## Example

```python
doe = rl.design(pct_treatment=[0.05, 0.10, 0.15], experiment_days=[21, 28, 35])

doe.plot_cluster_fits(scenario=0)
doe.plot_consolidated_fit(scenario=0)
doe.plot_power_analysis()
```
