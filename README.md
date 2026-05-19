# RealLift

<p align="center">
  <img src="https://raw.githubusercontent.com/RobertoJuniorWXYZ/RealLift/main/logo.png" width="200" style="border-radius: 10px;" alt="RealLift Logo">
</p>

**Causal Inference Framework for Geo Experiments & Marketing Science**

**RealLift** is an advanced Python library engineered to measure the true incremental impact (Lift) of marketing interventions through rigorous Causal Inference, Synthetic Control methodologies, and Robust Significance Testing.

---

## Framework Pillars

RealLift is built upon four layers of defense against noise and volatility:

1.  **Data Quality (Clean)**: Intelligent missing value imputation and filtering of noisy geographic entities to ensure high-fidelity inputs.
2.  **Auditable Planning (Design of Experiments)**: Algorithmically identifies the optimal geographic clusters and projects statistical power (MDE) prior to any campaign investment.
3.  **Causal Inference (Synthetic Control)**: Formulates robust counterfactuals via Convex Optimization with Intercept, ensuring behavioral alignment even across differing baseline levels.
4.  **Confidence Validation (Placebo & Significance)**: Defends analytical conclusions through Moving Block Bootstrap (MBB) and Permutation tests, providing indisputable visual and statistical evidence.

---

## Installation

```bash
pip install reallift
```

---

## Quick Start Guide

The RealLift 2.0 API is object-oriented. You instantiate a single `RealLift` orchestrator that holds the state of your data in-memory, making execution blazingly fast and extremely simple.

### 1. Initialize and Clean Data
Load your Pandas DataFrame and initialize the RealLift orchestrator.

```python
import pandas as pd
from reallift import RealLift

# Load your raw data
df = pd.read_csv("geo_daily_sales.csv")

# Initialize orchestrator
rl = RealLift(df, date_col="date")

# Clean, filter, and impute missing data (returns a clean DataFrame and a visual report)
df_clean = rl.clean(
    geos=None, # auto-detects
    imputation_method="interpolation"
)
```

### 2. Design the Experiment (Pre-Test Phase)
Use the DoE pipeline to rigorously select the best treatment and control geometries based on historical correlations.

```python
# Run the Design of Experiments
doe_results = rl.design_of_experiments(
    start_date="2025-01-01",
    end_date="2025-03-31",
    experiment_days=[21, 28, 30],
    n_folds=5
)

# Export an executive PDF report
doe_results.export_report("experiment_design.pdf")
```

### 3. Evaluate Incremental Impact (Post-Test Phase)
Measure the precise financial and percentual lift utilizing the Causal Inference engine against the final dataset after the campaign.

```python
# Assuming 'df_post' now contains the full historical + campaign data
rl = RealLift(df_post, date_col="date")

# Run the causal inference analysis
exp_results = rl.run(
    treatment_start_date="2025-04-01",
    treatment_end_date="2025-04-21",
    start_date="2025-01-01",
    doe=doe_results, 
    scenario=0
)
```

### 4. Professional Visual Diagnostics
RealLift ships with a "Black Premium" plotting suite designed for executive presentations.

```python
# View Treatment vs Synthetic for each cluster (Test period only)
exp_results.plot_cluster_effects(post_only=True)

# View aggregated causal impact across all regions
exp_results.plot_consolidated_effect(post_only=True)

# View the Hypothesis Test (H0 vs H1) using Bootstrap distributions
exp_results.plot_lift_distributions(show_null=True)
```

---

## Technical Differentiators

- **Object-Oriented & In-Memory**: Zero disk I/O bottlenecks. Manipulate and pass Pandas DataFrames directly into the engine.
- **Convex Intercept**: Intelligent baseline shift absorption that preserves the interpretability of synthetic weights ($\sum w = 1$).
- **Moving Block Bootstrap (MBB)**: Retains temporal autocorrelation during significance testing to prevent artificially tight confidence intervals.
- **Executive Aesthetics**: Built-in `matplotlib` dark themes tailored for C-Level data storytelling and unquestionable visual significance.

---

## Systems & Dependencies

- **Mathematics**: `cvxpy`, `scipy`, `numpy`
- **Data Engineering**: `pandas`, `scikit-learn`
- **Visualization**: `matplotlib`

---

Developed by **Roberto Junior**.