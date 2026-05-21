# ExperimentResult

Returned by [`GeoExperiment.run()`](GeoExperiment.run.md). Behaves as a regular `dict` and adds visualization methods.

```python
results = rl.run(...)
```

## Methods

### plot_cluster_effects

Treatment vs. synthetic control time series per cluster.

```python
results.plot_cluster_effects(post_only=False, figsize=(14, 4))
```

| Parameter | Default | Description |
|---|---|---|
| `post_only` | `False` | `True` shows only the treatment period; `False` shows full history |

### plot_consolidated_effect

Aggregated causal impact across all clusters.

```python
results.plot_consolidated_effect(post_only=False, figsize=(12, 5))
```

### plot_lift_distributions

Bootstrap confidence interval distributions with optional null hypothesis overlay.

```python
results.plot_lift_distributions(show_null=False, figsize=(15, 6))
```

| Parameter | Default | Description |
|---|---|---|
| `show_null` | `False` | Overlay the H₀ (null) distribution from the placebo bootstrap |

## Example

```python
results = rl.run(perform_backtesting={"lift": 0.0, "days": 28}, doe=doe, scenario=0)

results.plot_cluster_effects(post_only=False)
results.plot_consolidated_effect(post_only=False)
results.plot_lift_distributions(show_null=True)
```
