# RealLift

Base class. Handles data loading, date parsing, and geo detection. Subclass this to implement experiment-type-specific orchestrators.

```python
from reallift import RealLift
```

## Constructor

```python
RealLift(data, date_col, start_date=None, end_date=None, verbose=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `str \| DataFrame` | — | CSV file path or pandas DataFrame |
| `date_col` | `str` | — | Name of the date column. Values must be ISO format: `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS` |
| `start_date` | `str` | `None` | Optional window start (`"YYYY-MM-DD"`) |
| `end_date` | `str` | `None` | Optional window end (`"YYYY-MM-DD"`) |
| `verbose` | `bool` | `True` | Print initialization summary |

## Attributes

| Attribute | Description |
|---|---|
| `df` | Current working DataFrame (updated by subclass methods) |
| `geos` | List of geo column names |
| `date_col` | Name of the date column |
| `start_date` / `end_date` | Analysis window defaults |

## Subclasses

- [`GeoExperiment`](GeoExperiment.md) — geographic incrementality experiments
