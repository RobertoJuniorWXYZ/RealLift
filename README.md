# RealLift

<p align="center">
  <img src="https://raw.githubusercontent.com/RobertoJuniorWXYZ/RealLift/main/logo.png" width="200" style="border-radius: 10px;" alt="RealLift Logo">
</p>

**Causal Inference Framework for Geo Experiments & Marketing Science**

**RealLift** is an advanced Python library engineered to measure the true incremental impact (Lift) of marketing interventions through rigorous Causal Inference, Synthetic Control methodologies, and Robust Significance Testing.

---

## Framework Pillars

RealLift is built upon three layers of defense against noise and volatility:

1.  **Auditable Planning (Design of Experiments)**: Algorithmically identifies the optimal geographic clusters and projects statistical power (MDE) prior to any campaign investment.
2.  **Causal Inference (Synthetic Control)**: Formulates robust counterfactuals via Convex Optimization with Intercept, ensuring behavioral alignment even across differing baseline levels.
3.  **Confidence Validation (Placebo & Significance)**: Defends analytical conclusions through permutation tests based on the **MSPE Ratio** and non-parametric **Bootstrap** confidence intervals.

---

## Installation

```bash
pip install reallift
```

---

## Quick Start Guide

This end-to-end example demonstrates how to simulate geographic data, plan the optimal experiment design, inject a simulated marketing effect, and precisely measure the impact.

### 1. Generate Historical Data
Simulate daily performance for 27 candidate geos over a 90-day baseline.

```python
from reallift import generate_geo_data

geo_data = generate_geo_data(
    start_date="2025-01-01",
    end_date="2025-03-31",
    n_geos=27,
    pre_only=True,
    trend_slope=0.01,
    seasonality_amplitude=3,
    seasonality_period=7,
    noise_std=[1, 1.5],
    base_value=[50, 100],
    random_seed=42,
    save_csv=True,
    pre_file_name="demo_geodata_pre_test.csv"
)
```

### 2. Design the Experiment (Pre-Test Phase)
Use the DoE pipeline to rigorously select the best treatment and control geometries based on historical correlations.

```python
from reallift import design_of_experiments

doe = design_of_experiments(
    filepath="demo_geodata_pre_test.csv",
    date_col="date",
    start_date="2025-01-01",
    end_date="2025-03-31",
    pct_treatment=None,
    fixed_treatment=None,
    experiment_days=[21, 28, 30, 35, 60]
)
```

### 3. Inject Simulated Marketing Campaign
Simulate a 21-day campaign with an artificial lift applied strictly to the optimal treatment geos assigned in the DoE.

```python
from reallift import generate_simulated_intervention

geo_data_intervention = generate_simulated_intervention(
    filepath="demo_geodata_pre_test.csv",
    days=21,
    treatment_geos=doe['scenarios'][1]['treatment_pool'],
    lift=[0.05, 0.10],
    trend_slope=0.01,
    seasonality_amplitude=3,
    seasonality_period=7,
    noise_std=[1, 1.5],
    random_seed=42,
    save_csv=True,
    as_integer=True,
    file_name="demo_geodata_post_test.csv"
)
```

### 4. Evaluate Incremental Impact (Post-Test Phase)
Measure the precise financial and percentual lift utilizing the Causal Inference engine against the final dataset.

```python
from reallift import run_geo_experiment

results = run_geo_experiment(
    filepath="demo_geodata_post_test.csv",
    date_col="date",
    treatment_start_date="2025-04-01",
    treatment_end_date="2025-04-22",
    doe=doe, 
    scenario=1,
    plot=False
)
```

---

## Technical Differentiators

- **SER Engine (Synthetic Error Ratio)**: Proactive volatility filtering during the design phase to eliminate "Zombie Controls."
- **Convex Intercept**: Intelligent baseline shift absorption that preserves the interpretability of synthetic weights ($\sum w = 1$).
- **MSPE Ratio Strategy**: A placebo methodology resilient to the natural variance of high-frequency volatile markets.
- **Operational Freedom**: Aggressive donor curatorship via ElasticNet that often frees up to 50% of geographic regions for other commercial operations.

---

## Systems & Dependencies

- **Mathematics**: `cvxpy`, `scipy`, `numpy`
- **Data Engineering**: `pandas`, `scikit-learn`
- **Visualization**: `matplotlib`

---

Developed by **Roberto Junior**.