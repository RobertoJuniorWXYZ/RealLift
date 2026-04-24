# `reallift.simulation.generate_simulated_intervention`

Extrapolates a pre-existing database (CSV) to create a synthetic intervention period, injecting a lift over the projected behavior. It is the ideal tool to validate the **Statistical Power** of a DoE scenario before placing investment in the field.

## Signature

```python
def generate_simulated_intervention(
    filepath,
    treatment_geos,
    days=None,
    start_date=None,
    end_date=None,
    lift=0.05,
    date_col="date",
    noise_std=None,
    random_seed=42,
    plot=True,
    save_csv=False,
    file_name="simulated_intervention.csv",
    as_integer=False
) -> pd.DataFrame
```

---

## Operation (Weekday-Mean Forecast)

For each geography, the function builds the post-test period based on the **historical average per day of the week** (Monday, Tuesday, ..., Sunday). This naturally preserves weekly seasonality (retail behavior, weekends, etc.) without any complex parametric modeling.

**Stages per geo:**
1. Calculates the average of all pre-test records for each day of the week (0=Mon … 6=Sun)
2. Projects the future `days` using the respective weekday average as baseline
3. Adds Gaussian noise estimated from pre-test residuals (`std` per weekday)
4. Applies a zero floor (volumes cannot be negative)
5. Multiplies by the factor `(1 + lift)` in treated geos

---

## Parameters

### Input Data

| Parameter | Type | Description |
|:---|:---|:---|
| `filepath` | `str` | Path to the CSV with historical data (pre-test). |
| `date_col` | `str` | Name of the date column in the CSV. Default: `"date"`. |
| `treatment_geos` | `list` | List of geos that will receive the lift impact. |

### Post-Test Period Definition

Use **one** of the two ways below:

| Parameter | Type | Description |
|:---|:---|:---|
| `days` | `int` | Number of days to simulate starting from the last day of the pre-test. |
| `start_date` + `end_date` | `str` | Explicit dates for the post-test period (format `'YYYY-MM-DD'`, both inclusive). `days` is calculated automatically. |

> [!IMPORTANT]
> It is mandatory to provide `days` **or** the `start_date` + `end_date` pair. If neither is provided, the function will raise a `ValueError`.

### Effect Control

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `lift` | `float` or `[min, max]` | `0.05` | Relative incremental effect (e.g., `0.14` = +14%). If it's a list `[min, max]`, it draws a value within the range for each treated geo. |
| `noise_std` | `float` or `[min, max]` | `None` | Standard deviation of the additive noise. If `None`, automatically estimated from pre-test residuals. |

### Output

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `random_seed` | `int` | `42` | Seed for noise reproducibility. |
| `as_integer` | `bool` | `False` | If `True`, rounds all values to integers. |
| `plot` | `bool` | `True` | Displays the chart with the full period (pre + post-test). |
| `save_csv` | `bool` | `False` | Saves the full DataFrame to CSV. |
| `file_name` | `str` | `"simulated_intervention.csv"` | Name of the output file. |

---

## Return

Returns a `pd.DataFrame` containing the **union** of the original data (pre-test) with the simulated data (intervention), maintaining the same columns and the date column.

---

## Usage Examples

### Method 1 — by number of days

```python
from reallift import generate_simulated_intervention

df_sim = generate_simulated_intervention(
    filepath="cleaned_geo_data.csv",
    treatment_geos=["sao_paulo", "rio_de_janeiro", "campinas"],
    days=28,
    lift=0.14,
    date_col="dia",
    random_seed=42,
    save_csv=True,
    as_integer=True,
    file_name="lift_simulation.csv"
)
```

### Method 2 — by explicit dates

```python
from reallift import generate_simulated_intervention

df_sim = generate_simulated_intervention(
    filepath="cleaned_geo_data.csv",
    treatment_geos=["sao_paulo", "rio_de_janeiro", "campinas"],
    start_date="2026-01-01",
    end_date="2026-04-10",   # 100 days, inclusive
    lift=0.14,
    date_col="dia",
    random_seed=42,
    save_csv=True,
    as_integer=True,
    file_name="lift_simulation.csv"
)
```

---

## Relation to DoE

The standard flow is to use `generate_simulated_intervention` right after `design_of_experiments`, feeding the geos selected by the best scenario directly:

```python
# Best DoE scenario
best_scenario = doe["scenarios"][0]

df_sim = generate_simulated_intervention(
    filepath="cleaned_geo_data.csv",
    treatment_geos=best_scenario["treatment_pool"],
    days=28,
    lift=0.14,
    date_col="dia",
    as_integer=True,
    save_csv=True,
    file_name="lift_simulation.csv"
)
```
