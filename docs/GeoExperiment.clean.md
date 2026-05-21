# GeoExperiment.clean

Filter and impute the geo data. Updates `self.df` in place and returns the cleaned DataFrame.

```python
rl.clean(
    imputation_method="constant",
    constant_value=1e-3,
    max_zero_rate=None,
    top_n_geos=None,
    keep_top_quantiles=None,
    quantile_bins=None,
    exclude_geos=None,
    save_csv=True,
    file_name="cleaned_geo_data.csv",
    save_pdf=False,
    pdf_name="cleaning_report.pdf",
    plot=False,
    verbose=None,
    logo=None,
)
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `imputation_method` | `str` | `"constant"` | `"constant"` fills with `constant_value`; `"interpolation"` uses linear infill |
| `constant_value` | `float` | `1e-3` | Floor value for constant imputation — prevents `log(0)` downstream |
| `max_zero_rate` | `float` | `None` | Drop geos with zero/missing rate above this threshold (0.0–1.0) |
| `top_n_geos` | `int` | `None` | Keep only the top N geos by quality-volume ranking |
| `keep_top_quantiles` | `int` | `None` | Keep geos in the top N volume quantiles |
| `quantile_bins` | `int` | `None` | Number of quantile bins for zero-rate segmentation (Q1 = cleanest geos) |
| `exclude_geos` | `list` | `None` | Hard-exclude these geo names regardless of quality |
| `save_csv` | `bool` | `True` | Write cleaned data to `file_name` |
| `file_name` | `str` | `"cleaned_geo_data.csv"` | Output CSV path |
| `save_pdf` | `bool` | `False` | Export a data quality PDF report |
| `pdf_name` | `str` | `"cleaning_report.pdf"` | Output PDF path |
| `plot` | `bool` | `False` | Show before/after matplotlib chart for the top 20 geos |
| `logo` | `str` | `None` | Path to a logo image to embed in the PDF report |
| `verbose` | `bool` | `None` | Override instance verbosity for this call |

## Returns

`pd.DataFrame` — cleaned DataFrame with the date column and surviving geo columns.

## Example

```python
df_clean = rl.clean(
    max_zero_rate=0.0,
    keep_top_quantiles=1,
    quantile_bins=40,
    save_pdf=True,
    pdf_name="DataQualityReport.pdf",
)
```
