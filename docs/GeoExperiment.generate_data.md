# GeoExperiment.generate_data

Generate synthetic geo time-series data for testing and prototyping. Class method — no instance required.

```python
from reallift import RealLift
```

## Signature

```python
RealLift.GeoExperiment.generate_data(
    start_date="2020-01-01",
    end_date="2020-03-31",
    n_geos=27,
    freq="D",
    mean_values=[100, 500],
    trend_slope=0.00,
    seasonality_amplitudes=0.10,
    seasonality_period=7,
    noise_std=0.05,
    random_seed=42,
    n_zeros=0,
    as_integer=True,
    plot=True,
    save_csv=False,
    file_name="synthetic_geo_data.csv",
)
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start_date` | `str` | `"2020-01-01"` | Start of the generated period |
| `end_date` | `str` | `"2020-03-31"` | End of the generated period |
| `n_geos` | `int` | `27` | Number of geographies to generate |
| `freq` | `str` | `"D"` | Pandas date frequency |
| `mean_values` | `float \| list` | `[100, 500]` | Mean volume per geo. Pass `[min, max]` to give each geo a different random mean — useful for simulating markets with different volume scales |
| `trend_slope` | `float` | `0.00` | Linear trend coefficient |
| `seasonality_amplitudes` | `float \| list` | `0.10` | Amplitude as a fraction of each geo's mean. `0.10` = ±10% oscillation. Pass `[min, max]` to randomize per geo |
| `seasonality_period` | `int` | `7` | Period of the seasonal component (days) |
| `noise_std` | `float \| list` | `0.05` | Noise as a fraction of each geo's mean. `0.05` = ~5% random variation. Pass `[min, max]` to randomize per geo |
| `random_seed` | `int` | `42` | Seed for reproducibility |
| `n_zeros` | `int` | `0` | Number of zero-value holes to inject randomly |
| `as_integer` | `bool` | `True` | Round output to integers |
| `plot` | `bool` | `True` | Show the generated data chart |
| `save_csv` | `bool` | `False` | Write dataset to `file_name` |
| `file_name` | `str` | `"synthetic_geo_data.csv"` | Output CSV path |

## Returns

`pd.DataFrame` — date column plus one column per geo.

## Example

```python
df = RealLift.GeoExperiment.generate_data(
    start_date="2020-01-01",
    end_date="2020-03-31",
    n_geos=27,
    mean_values=[100, 500],
    seasonality_amplitudes=0.10,
    seasonality_period=7,
    noise_std=0.05,
    random_seed=42,
    as_integer=True,
    plot=True,
)
```
