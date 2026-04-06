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

### 1. Planning (Pre-Test Phase)
Use the DoE pipeline to project the duration strictly necessary to capture a target Minimum Detectable Effect (MDE).

```python
from reallift.pipelines import design_of_experiments

# Project requirements and identify optimal clusters
doe_result = design_of_experiments(
    filepath="historical_data.csv",
    date_col="date",
    pct_treatment=0.10,      # Test a scenario with 10% of the market treated
    mde=0.015,               # Target 1.5% incremental lift
    experiment_days=[21, 60] # Evaluation window between 21 and 60 days
)
```

### 2. Impact Analysis (Post-Test Phase)
Execute the complete analytical pipeline after or during the intervention.

```python
from reallift.pipelines import run_geo_experiment

# Complete end-to-end analytical pipeline
result = run_geo_experiment(
    filepath="experiment_data.csv",
    date_col="date",
    treatment_start_date="2025-05-01",
    doe=doe_result,          # Inherit validated clusters from DoE
    scenario=0               # Use the 10% treatment scenario
)

# Extract total documented incremental impact
print(f"Total Cumulative Lift: {result['results'][0]['synthetic']['lift_total']:.2f}")
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