# RealLift

<p align="center">
  <img src="https://raw.githubusercontent.com/RobertoJuniorWXYZ/RealLift/main/logo.png" width="200" style="border-radius: 10px;" alt="RealLift Logo">
</p>

**Causal Inference Library for Lift Measurement & Design of Experiments**

RealLift is an advanced Python library engineered to assist data scientists and analysts in reliably measuring the true incremental impact of interventions through rigorous causal inference methodologies, such as **GeoLift**, **Synthetic Control Estimation**, and **Placebo Testing**.

---

## Capabilities

- **Design of Experiments (Geo-Splitting)**: Algorithmically identifies structural clusters and mathematically selects the optimal treatment and control regions based on ElasticNet feature selection and convex proximity matrices.
- **Synthetic Control Measurement**: Formulates robust counterfactual interventions by mapping temporal correlations across a predefined array of donor regions via constrained Convex Optimization (`cvxpy`).
- **Time Series Cross-Validation**: Ensures predictive validity of counterfactuals via Historical Simulation, isolating definitive Out-Of-Fold (OOF) $R^2$ and MAPE limits prior to experiments.
- **Duration & Statistical Power**: Estimates predictive power dynamically over time streams to establish strict Minimum Detectable Effect (MDE) bounds before test implementation.
- **Significance & Placebo Testing**: Empirically defends the analytical conclusions through non-parametric bootstrap sampling and randomized spatial placebo permutations to comprehensively evaluate the null hypothesis.

---

## Installation

RealLift is securely distributed through **PyPI** for production environments:

```bash
pip install reallift
```

Alternatively, obtain the latest development snapshot directly from the source repository:

```bash
pip install git+https://github.com/RobertoJuniorWXYZ/RealLift.git
```

---

## Quick Start Guide

### 1. Requirements & Design (Pre-Test Phase)
Before executing a field intervention, analyze the underlying baseline correlation to discover optimal clusters and project the duration strictly necessary to capture a target Minimum Detectable Effect (MDE).

```python
from reallift import run_geo_requirements

# Identify structural blocks and validate exposure durations
summary = run_geo_requirements(
    filepath="historical_data.csv",
    date_col="date",
    n_treatment=1,
    mde=0.015,
    max_days=[21, 60],
    n_folds=5,
    verbose=True
)
```

### 2. Intervention Measurement (Post-Test Phase)
Following the completion of an intervention, apply the algorithmic pipeline encompassing validation constraint-checking, Synthetic Control extraction, and empirical Placebo diagnostics.

```python
from reallift import run_geo_experiment

# Execute the complete analytical pipeline
result = run_geo_experiment(
    filepath="experiment_data.csv",
    date_col="date",
    treatment_start_date="2025-05-01",
    n_treatment=1,
    mde=0.015,
    max_days=[21, 60],
    n_folds=5,
    random_state=42,
    verbose=True
)

# Extract total absolute impact estimates
print(f"Incremental Lift (abs): {result['results'][0]['synthetic']['lift_total']:.2f}")
```

---

## Examples & Application

For a comprehensive methodological demonstration concerning correlation assumptions, feature engineering operations, and diagnostic evaluation limits, refer to the Jupyter notebooks mapped under the `examples/geotests/` directory within the primary repository.

---

## Systems & Dependencies

- **Platform Target**: Python 3.8+
- **Mathematics**: `cvxpy` (Core algorithmic solver for constraints)
- **Data Engineering**: `pandas`, `numpy`, `scikit-learn`, `scipy`
- **Plotting Engines**: `matplotlib`, `seaborn`

---

## License

MIT License. Navigate to the `LICENSE` file for full disclosure.

---

Developed by **Roberto Junior**.