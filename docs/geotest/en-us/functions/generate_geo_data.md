# `reallift.simulation.generate_geo_data`

Generates synthetic databases for geographic tests, allowing simulation of complex trend, seasonality, and noise scenarios. It is the primary tool to validate model sensitivity (MDE) before running real experiments.

## Signature

```python
def generate_geo_data(
    start_date="2022-01-01",
    end_date="2022-06-30",
    n_geos=5,
    freq="D",
    trend_slope=0.05,
    seasonality_amplitude=10,
    seasonality_period=7,
    noise_std=2,
    treatment_geos=None,
    treatment_start=None,
    lift=0.2,
    random_seed=42,
    plot=True,
    save_csv=False,
    file_name="synthetic_geolift.csv",
    pre_file_name="synthetic_geolift_pre.csv",
    base_value=50.0,
    as_integer=False,
    pre_only=False
) -> tuple
```

## Main Parameters

- **`n_geos`**: Quantity of geographic units to be generated.
- **`trend_slope`**: Linear trend slope (organic growth).
- **`seasonality_amplitude`**: Magnitude of sinusoidal oscillation (e.g., weekly cycles).
- **`noise_std`**: Standard deviation of white noise inserted in series. Can be a `float` or a list `[min, max]` for per-geo randomness.
- **`lift`**: Incremental effect to be injected into treatment geos. Can be a fixed value (0.05 = 5%) or a range for random drawing.
- **`pre_only`**: (New) If `True`, function generates and returns only the "pre-test" period, ignoring lift injection and future dates. Essential for **Design of Experiments (DoE)** workflows.

## Return

Returns a tuple `(df_full, df_pre, treatment_geos)`:
1. `df_full`: Full DataFrame with all generated period.
2. `df_pre`: Filtered DataFrame with only the period before `treatment_start`.
3. `treatment_geos`: List of geos that were (or would be) drawn for treatment.

## Usage Example

```python
from reallift.simulation import generate_geo_data

# Generating pre-test data for a quick DoE
df, df_pre, treated = generate_geo_data(
    n_geos=30,
    noise_std=[2, 8],
    pre_only=True,
    save_csv=True,
    pre_file_name="my_history.csv"
)
```
