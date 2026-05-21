# RealLift

<p align="center">
  <img src="https://raw.githubusercontent.com/RobertoJuniorWXYZ/RealLift/main/logo.png" width="200" style="border-radius: 10px;" alt="RealLift Logo">
</p>

<a href="https://doi.org/10.5281/zenodo.20329451"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20329451.svg" alt="DOI"></a>


**Causal Inference Framework for Geo Experiments & Marketing Science**

**RealLift** is a Python library for measuring the true incremental impact of marketing interventions through Synthetic Control, Convex Optimization, and robust significance testing — designed for markets where individual-level tracking is unavailable.

---

## Framework Pillars

Four layers of defense against noise and spurious correlations:

1. **Data Quality (`clean`)** — Missing value imputation, zero-rate filtering, and geo-level quality scoring.
2. **Auditable Planning (`design`)** — Algorithmically selects optimal treatment/control clusters and projects statistical power (MDE) before any spend.
3. **Causal Inference (`run`)** — Builds counterfactuals via Convex Optimization with Intercept Adjustment, ensuring behavioral alignment across differing baseline levels.
4. **Confidence Validation** — Moving Block Bootstrap (MBB) + Placebo/Permutation tests with visual and statistical evidence.

---

## Installation

```bash
pip install reallift
```

---

## Quick Start

The API is object-oriented. Instantiate a `GeoExperiment` via the `RealLift` namespace — it holds the data in memory across all pipeline steps.

### 1. Initialize

```python
from reallift import RealLift

# Accepts a CSV path or a pandas DataFrame
rl = RealLift.GeoExperiment(
    "geo_daily_sales.csv",
    date_col="date",
    start_date="2025-01-01",   # optional: window the analysis period
    end_date="2025-04-30",
)
```

### 2. Clean

```python
df_clean = rl.clean(
    imputation_method="constant",   # "constant" or "interpolation"
    max_zero_rate=0.0,              # drop geos with any zero-revenue days
    keep_top_quantiles=1,           # keep only top revenue quantile
    quantile_bins=40,
    save_pdf=True,
    pdf_name="DataQualityReport.pdf",
)
```

### 3. Design (Pre-Test Phase)

```python
doe = rl.design(
    pct_treatment=[0.05, 0.10, 0.15],   # treatment pool sizes to evaluate
    experiment_days=[21, 28, 35],        # durations to evaluate
    check_ghost_lift=True,               # OOS backtest for spurious effects
    save_pdf=True,
    pdf_name="DoEReport.pdf",
)

# Visual diagnostics
doe.plot_cluster_fits(scenario=0)       # pre-period synthetic control fit per cluster
doe.plot_consolidated_fit(scenario=0)  # aggregated control group fit
doe.plot_power_analysis()              # MDE curves vs. experiment duration
```

### 4. Run (Post-Test Phase)

**Option A — Backtest on historical data** (validate the design before the campaign):

```python
results = rl.run(
    perform_backtesting={"lift": 0.0, "days": 28},  # placebo: inject 0% lift
    doe=doe,
    scenario=0,
)
```

**Option B — Real campaign analysis** (after the campaign ends, load data that includes the treatment period):

```python
rl_post = RealLift.GeoExperiment("geo_daily_sales_with_campaign.csv", date_col="date")
rl_post.clean()

results = rl_post.run(
    treatment_start_date="2025-04-01",
    treatment_end_date="2025-04-28",
    doe=doe,
    scenario=0,
)
```

### 5. Visualize Results

```python
results.plot_cluster_effects(post_only=False)      # treatment vs. synthetic per cluster
results.plot_consolidated_effect(post_only=False)  # aggregated causal impact
results.plot_lift_distributions(show_null=True)    # bootstrap CI + hypothesis test
```

---

> **Hands-on example:** [`examples/GeoExperiment/GeoExperiment-Quickstart.ipynb`](examples/GeoExperiment/GeoExperiment-Quickstart.ipynb) — full pipeline with synthetic data, no setup required.

---

## Key Algorithms

| Algorithm | Role |
|---|---|
| **Convex Intercept** | Absorbs baseline shift between treated and synthetic while preserving Σw = 1 |
| **Synchronization Error Ratio (SER)** | Pre-experiment geo ranking and volatility scoring |
| **ElasticNet Pool Purification** | Removes weak/spurious controls before convex optimization |
| **Moving Block Bootstrap (MBB)** | Respects temporal autocorrelation (~7-day blocks) for conservative CIs |
| **OOS Ghost Lift Detection** | Weekly-aligned backtest that flags spurious effects in the experiment design |
| **Placebo / MSPE Ratio** | Permutation-based empirical p-values robust to noisy markets |

---

## Dependencies

- **Optimization**: `cvxpy`, `scipy`, `numpy`
- **Data**: `pandas`, `scikit-learn`
- **Visualization**: `matplotlib`

---

Developed by **Roberto Junior**.

---

## Citation

If you use RealLift in your research or applied work, please cite:

```bibtex
@misc{junior2026reallift,
  author    = {Junior, Roberto},
  title     = {RealLift: Robust Design and Inference for Geo-Experimentation},
  year      = {2026},
  doi       = {10.5281/zenodo.20328611},
  url       = {https://doi.org/10.5281/zenodo.20328611},
  note      = {Preprint}
}
```
