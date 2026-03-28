# 🚀 RealLift

**Causal Inference Library for Lift Measurement**

RealLift is a powerful Python library designed to help data scientists and marketers measure the true impact of their interventions (treatments) using advanced causal inference techniques, such as **GeoLift**, **Synthetic Control**, and **Placebo Testing**.

---

## ✨ Features

- **🎯 Geo-Splitting**: Automatically find the best treatment and control groups based on historical correlation.
- **📈 Synthetic Control**: Build counterfactuals using a weighted combination of control regions.
- **🧪 Significance Testing**: Robust bootstrap-based confidence intervals and p-values.
- **🛡️ Placebo Tests**: Validate your model's reliability by running "fake" experiments on control groups.
- **⏳ Duration Estimation**: Calculate the necessary experiment duration based on MDE (Minimum Detectable Effect).
- **📊 Professional Visualizations**: Built-in plotting for experiment results and validation.

---

## 🚀 Installation

Install the stable version via **PyPI**:

```bash
pip install reallift
```

Or install the latest development version directly from **GitHub**:

```bash
pip install git+https://github.com/RobertoJuniorWXYZ/RealLift.git
```

---

## ⚡ Quick Start

### 1. Requirements & Design
Before starting an experiment, find the best clusters and estimate the required duration for a target MDE.

```python
from reallift import run_geo_requirements

# Find best geo clusters and estimate duration
summary = run_geo_requirements(
    filepath="historical_data.csv",
    date_col="date",
    n_treatment=1,
    mde=0.015,
    max_days=60
)
```

### 2. Run a Complete Geo Experiment
Execute the full pipeline including validation, duration estimation, synthetic control, and placebo tests.

```python
from reallift import run_geo_experiment

# Run full experiment pipeline
result = run_geo_experiment(
    filepath="your_data.csv",
    date_col="date",
    treatment_start_date="2025-05-01",
    n_treatment=1,
    mde=0.02
)

# Access results
print(f"Observed Lift: {result['results'][0]['synthetic']['lift_mean_abs']:.4f}")
```

---

## 📖 Examples

For a deep dive into the library's capabilities, check out our demonstration notebooks in the [examples/geotests/](examples/geotests/) directory.

---

## 🛠️ Requirements

- Python 3.8+
- Pandas, Numpy, Scikit-Learn, Scipy, Matplotlib, CVXPY

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

Developed with ❤️ by **Roberto Junior**