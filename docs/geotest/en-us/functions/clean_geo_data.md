# `reallift.utils.data_cleaning.clean_geo_data`

The `clean_geo_data` function acts as the **DataOps Portal** of the RealLift framework. Its purpose is to prepare, standardize, and validate raw geospatial data, ensuring that time series are statistically robust for Causal Inference algorithms (Synthetic Control and Log-Diff).

## Signature

```python
def clean_geo_data(
    data, 
    date_col: str, 
    imputation_method: str = 'interpolation', 
    constant_value: float = 1e-3, 
    verbose: bool = True,
    plot: bool = False,
    save_csv: bool = True,
    save_pdf: bool = False,
    file_name: str = 'cleaned_geo_data.csv',
    pdf_name: str = 'cleaning_report.pdf',
    max_zero_rate: float = None,
    top_n_geos: int = None,
    keep_top_quantiles: int = None,
    exclude_geos: list = None,
    quantile_bins: int = None,
    start_date: str = None,
    end_date: str = None,
    logo: str = None
) -> pd.DataFrame
```

---

## 1. Pipeline Operation

The cleaning pipeline executes six sequential stages to transform noisy revenue/sales data into mathematical matrices ready for modeling:

1.  **Temporal Standardization:** Converts various date formats to the ISO standard (YYYY-MM-DD).
2.  **Chronological Ordering:** Reorganizes the timeline and detects missing day gaps.
3.  **Geospatial Segregation:** Separates the temporal dimension from the metrics of each geography.
4.  **Algebraic Imputation (Anti-Log Crash):** Treats zero or empty (NaN) cells to avoid the $-\infty$ mathematical error during the algorithm's internal logarithmic transformations.
5.  **Quality Scorecard:** Generates a detailed diagnostic on the sparsity and volume distribution of each city.
6.  **Strategic Filtering:** Dynamically removes low-quality or low-volume cities based on business rules (Pareto, Zero Rate, or Ranking).

---

## 2. Parameters

### 2.1 Data and Time Configuration

| Parameter | Type | Description |
|:---|:---|:---|
| `data` | `DataFrame` / `str` | Data source: a Pandas DataFrame or the path to a CSV file. |
| `date_col` | `str` | Name of the column representing the temporal dimension (dates). |
| `exclude_geos` | `list` | List of geography names for forced exclusion (known outliers or test markets). |
| `start_date` | `str` | *(Optional)* Filters the dataset to include only records **from this date** (inclusive). Format: `'YYYY-MM-DD'`. |
| `end_date` | `str` | *(Optional)* Filters the dataset to include only records **up to this date** (inclusive). Format: `'YYYY-MM-DD'`. |

> [!TIP]
> **Period Filtering:** Use `start_date` and `end_date` to restrict analysis to a specific temporal window within a larger dataset. For example, if the CSV covers 2 years but you want to prepare only a 90-day pre-test for the DoE, set `end_date` to the campaign's start date.

### 2.2 Imputation Strategy

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `imputation_method` | `str` | `'interpolation'` | `'interpolation'`: Fills gaps via linear interpolation (tends to preserve trend). <br> `'constant'`: Fills all blanks with the `constant_value`. |
| `constant_value` | `float` | `1e-3` | The "floor" value injected. Essential to avoid `log(0)` errors. |

### 2.3 Selection and Filtering Mechanisms (DataOps)

| Parameter | Type | Description |
|:---|:---|:---|
| `max_zero_rate` | `float` | Removes geos with a zero/empty rate higher than the limit (e.g., `0.2` removes cities with >20% zeroed days). |
| `quantile_bins` | `int` | Number of slices for volume distribution analysis (e.g., `4` for Quartiles, `10` for Deciles). |
| `keep_top_quantiles`| `int` | Keeps only the top N quantiles (e.g., `1` keeps only Q1 — the highest volume cities). |
| `top_n_geos` | `int` | Selects the top N geographies based on a combined Volume and Quality ranking. |

### 2.4 Outputs and Report

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `save_csv` | `bool` | `True` | Exports the resulting DataFrame to a CSV. |
| `file_name` | `str` | `'cleaned_geo_data.csv'` | Name of the output CSV file. |
| `save_pdf` | `bool` | `False` | Generates an audit PDF report (`cleaning_report.pdf`). |
| `pdf_name` | `str` | `'cleaning_report.pdf'` | Name of the output PDF file. |
| `logo` | `str` | `None` | Path to a logo to be included in the PDF header. |

---

## 3. Understanding the Diagnostic Scorecard

When the `verbose=True` parameter is used, the function prints the **Geos Scorecard** in the terminal. This table is vital for auditing the impact of cleaning on your data:

| Column | Meaning | Interpretation |
|:---|:---|:---|
| **Geo** | Market Name | Unit identification. |
| **Imputed** | N of treated cells | How many days were "invented" or corrected via imputation. |
| **% Zeros** | Sparsity Rate | Proportion of days the city had no sales/revenue. |
| **Sum Original** | Raw Volume | Total sum of values before cleaning. |
| **% Imputed** | Volumetric Impact | How much of the city's final volume is due to imputation. |

> [!IMPORTANT]
> **The Golden Rule of Imputation**
> If a city has `% Imputed` above 5%, Synthetic Control weights may be distorted by over-synthesized data. It is recommended to filter these markets using `max_zero_rate`.

---

## 4. Quantile Analysis (Pareto Principle)

RealLift uses volume segmentation to help choose the donor pool. In the terminal, you will see the quantile analysis:

- **Q1 (Top Volume):** Generally capitals and large centers. They are great treatments but dangerous donors if too dominant.
- **Q4/Q10 (Long Tail):** Small and noisy cities. They should be filtered in the DoE to prevent the algorithm from trying to polish error using irrelevant cities.

---

## 5. Usage Example

The standard DataOps workflow before starting an experiment is:

```python
from reallift.utils.data_cleaning import clean_geo_data

# 1. Cleaning and Filtering: Keeping only high-volume (Q1 and Q2) and stable (<10% zeros) markets
#    restricted to the pre-test period
df_clean = clean_geo_data(
    data="raw_sales.csv",
    date_col="date",
    start_date="2024-01-01",     # Filter from this date
    end_date="2025-12-31",       # Filter until this date (inclusive)
    max_zero_rate=0.10,          # Removes geos with more than 10% zeroed days
    quantile_bins=5,             # Divides volume into Quintiles
    keep_top_quantiles=2,        # Keeps only Q1 and Q2 (Top 40% in volume)
    imputation_method='interpolation'
)

# 2. The 'df_clean' result is now ready for DoE
# clusters = design_of_experiments(data=df_clean, ...)
```

---

## 6. Forensic Visualization (`plot=True`)

When `plot=True` is activated, the function generates a **"Before vs After"** comparative chart for the top 20 geographies. This allows your data team to visually validate whether the interpolation strategy is preserving seasonal trends or creating unwanted artificial patterns.
