# GeoExperiment

Geographic incrementality experiment orchestrator. Inherits data loading from [`RealLift`](RealLift.md).

```python
from reallift import RealLift
```

## Constructor

```python
RealLift.GeoExperiment(data, date_col, start_date=None, end_date=None, verbose=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `str \| DataFrame` | — | CSV file path or pandas DataFrame |
| `date_col` | `str` | — | Name of the date column. Values must be ISO format: `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS` |
| `start_date` | `str` | `None` | Optional window start (`"YYYY-MM-DD"`) |
| `end_date` | `str` | `None` | Optional window end (`"YYYY-MM-DD"`) |
| `verbose` | `bool` | `True` | Print initialization summary |

## Methods

- [`clean()`](GeoExperiment.clean.md)
- [`design()`](GeoExperiment.design.md)
- [`run()`](GeoExperiment.run.md)
- [`generate_data()`](GeoExperiment.generate_data.md) — `@classmethod`

## Example

```python
rl = RealLift.GeoExperiment(
    "geo_daily_sales.csv",
    date_col="date",
    start_date="2025-01-01",
    end_date="2025-04-30",
)
rl.clean()
doe = rl.design(pct_treatment=[0.05, 0.10], experiment_days=[21, 28])
results = rl.run(perform_backtesting={"lift": 0.0, "days": 28}, doe=doe, scenario=0)
```
