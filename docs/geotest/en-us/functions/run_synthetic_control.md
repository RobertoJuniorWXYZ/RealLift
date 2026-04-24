# `reallift.geo.synthetic.run_synthetic_control`

The `run_synthetic_control` function is the causal inference core of RealLift. It uses Convex Optimization to build a counterfactual (Synthetic Control) that mimics treatment unit behavior in the period before intervention, allowing the calculation of real incremental impact.

## Signature

```python
def run_synthetic_control(
    filepath: str,
    date_col: str,
    treatment_geo: str | list,
    control_geos: list,
    treatment_start_date: str,
    treatment_end_date: str = None,
    start_date: str = None,
    end_date: str = None,
    random_state: int = None,
    cluster_idx: int = None,
    plot: bool = True,
    verbose: bool = True
) -> dict
```

## Mathematical and Algorithmic Foundation

### 1. Convex Optimization (Pre-Test Period)
The algorithm isolates data before `treatment_start_date` and seeks weights $w$ and intercept $\alpha$ that minimize squared error between actual and synthetic:

$$ \min_{w, \alpha} \| Y_{pre} - (X_{pre} \cdot w + \alpha) \|^2 $$

**Convex Constraints:**
- All weights are non-negative ($w_i \ge 0$).
- Sum of weights is exactly 1 ($\sum w = 1$).

The **Convex Intercept ($\alpha$)** is a pragmatic innovation that absorbs systematic level differences (different average sales), allowing weights to purely focus on aligning series *correlation* and *behavior*.

### 2. Counterfactual Projection (Post-Test Period)
Once ideal weights are found, they are applied to intervention period data. **Lift** is the cumulative difference between what geography actually sold ($Y_{post}$) and what the model predicted it would sell without marketing ($\hat{Y}_{post}$):

$$ \text{Lift} = \sum (Y_{post} - \hat{Y}_{post}) $$

### 3. Significance via Bootstrap
Instead of depending on rigid parametric assumptions, RealLift uses **re-sampling (Bootstrap)** on historical residuals to generate 95% confidence intervals and determine if observed Lift is statistically different from zero.

## Return (*Output*)

Returns a dictionary with all experiment intelligence:

```python
{
    "weights": {"geo_A": 0.6, "geo_B": 0.4}, # Donor convex weights
    "alpha": 12.5,                           # Intercept value (level correction)
    "lift_total": 5212.0,                    # Total cumulative incremental impact
    "lift_mean_pct": 0.045,                  # Mean percentage lift (4.5%)
    "pre_mspe": 0.0012,                      # Mean squared error in pre-test
    "post_mspe": 0.2540,                     # Mean squared error in post-test
    "mspe_ratio": 211.6,                     # Post/Pre ratio (robustness metric)
    "bootstrap": { ... },                    # Confidence intervals and boot p-value
    "df": pd.DataFrame,                      # Processed database with actual and synthetic
    "plotting_data": { ... }                 # Structures ready for visualization
}
```
