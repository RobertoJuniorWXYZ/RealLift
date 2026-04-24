# `reallift.geo.did.run_matched_did`

The `run_matched_did` function composes the "twin half" of RealLift's causal control. Rigorously designed on the original foundations of Difference-in-Differences (DiD), this methodological path leverages computational power while bypassing *Convex Synthetic Control* limitations.

By abandoning dynamic continuous weights in favor of **perfectly uniform average (`[1/N]`)**, DiD protects inferences executed on highly unstable baselines (where L2 matrix optimization would fail via Variance Suppression).

## Signature

```python
def run_matched_did(
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

## Mathematical and Algorithmic Foundation (Panel Data)

The classic "Volume Discrepancy" problem between regions doesn't hinder time series in RealLift due to the excellent internal algorithmic system known as **Unit Fixed Effect** (Normalization).

### 1. Leveled Vector Construction
When operating continuous time series (Event-Study), donor curves rarely have the same scale mass as the main Treated Market.

The algorithm initially divides data by pre-intervention period mean to recover not "sales", but the *market oscillation* curve.

$$
X_{norm} = \frac{X_{pre}}{\bar{X}_{pre\_mean}}
$$

### 2. Interception and Baseline (*Parallel Trends*)

While the Convex path needs to strictly discover complex weights, the differential mode defines **Homogeneous Weights (Control Group Axiom)**:

$$ w = \left[ \frac{1}{N_{\text{controls}}}, \dots, \frac{1}{N_{\text{controls}}} \right] $$

And magically projects the market index to the isolated base volumetric proportions of the original treated site, ensuring both start at the same $\alpha$ intercept before treatment date causes graph rupture!

$$
Baseline = \left( \sum (X_{norm} \cdot w) \right) \cdot \bar{Y}_{pre\_mean}
$$

### 3. Estimated Relative Impact and Resampling

Like Synthetic Control, every absolute Lift measured day-against-day in post-test is analyzed via **Empirical Resampling (Iterative Bootstrap)** against the neutral noise trend. Without involving traditional theoretical statistics Gaussian equations.

## Return (*Output*)

API abstraction is kept structurally **identical** to `run_synthetic_control` return to allow clean routing of the `design_of_experiments` abstract pipeline.

```python
{
    "weights": {"geo_B": 0.5, "geo_C": 0.5}, # Classic DiD (uniform 1/N)
    "alpha": 0.0,                            # Directly normalized
    "lift_total": 451.2,                     # Accumulated incremental impact
    "lift_mean_pct": 0.065,                  # Mean percentage lift (6.5%)
    "pre_mspe": 0.051,                       # Prediction error on previous trend (Panel)
    "post_mspe": 0.982,                      # Rupture distance at intervention
    "mspe_ratio": 19.25,                     # Post/Pre Placebo Allowance
    "bootstrap": { ... },                    # Confidence Intervals
    "df": pd.DataFrame,                      # Consolidated post-transformation matrix
    "plotting_data": { ... }                 # Facilitates visuals decoupled from verbose
}
```
