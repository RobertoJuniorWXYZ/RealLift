# `reallift.geo.discovery.discover_geo_clusters`

In the RealLift library, the `discover_geo_clusters` function is the primary engine for experimental design of geography-based incremental tests (Geo Experiments). Its purpose is to predict and identify the optimal combination of control sub-regions to form a **Synthetic Control** with high correlation and low stochastic error relative to a target treatment region, **before the intervention period.**

## Signature

```python
def discover_geo_clusters(
    filepath: str,
    date_col: str,
    geos: list = None,
    n_treatment: int = 3,
    fixed_treatment: list = None,
    start_date: str = None,
    end_date: str = None,
    use_elasticnet: bool = True,
    search_mode: str = "auto",
    alpha: float | list = 0.01,
    l1_ratio: float | list = 0.5,
    n_jobs: int = None,
    verbose: bool = True,
    show_results: bool = True
) -> list
```

---

## 1. Objective

The function solves the following combinatorial optimization problem applied to Causal Inference:

> **Given a set of $n$ geographic time series observed in the pre-intervention period, partition these geographies into a treatment group $(T)$ and a control/donor group $(D)$ such that the Synthetic Control constructed from $D$ reproduces the historical trajectory of $T$ with maximum fidelity.**

The quality of this partition is the determining factor for the credibility of the entire subsequent incremental test.

The problem is non-trivial because:
1. The search space is combinatorial: $C(n, k)$ possible partitions grow factorially.
2. Synthetic Control construction involves nested constrained convex optimization within each evaluation.
3. The ranking metric must balance two potentially conflicting criteria (low error vs. high correlation).

---

## 2. Inputs

The function receives tabular data in CSV format with the following expected structure:

### 2.1 CSV Structure

| Column | Description | Type | Constraints |
|:---|:---|:---|:---|
| `date_col` | Temporal indexing column | `str` / `datetime` | Parseable as a date; `YYYY-MM-DD` or `DD/MM/YYYY` format |
| `geo_1, geo_2, ..., geo_n` | One column per geography | `float` / `int` | Values ≥ 0 (required for logarithmic transformation) |

### 2.2 Formal Data Requirements

- **Temporal granularity:** Daily (the function aggregates duplicates via `groupby(date_col).sum()`).
- **Completeness:** All geographies must have observations on all dates. Missing values result in `NaN` after log-diff transformation and cause silent optimization failure.
- **Strict positivity:** Since the first transformation applied is $\log(x)$, zero or negative values produce $-\infty$ or `NaN`. If a geography has zero-value days, it must be excluded or imputed before the call.
- **Minimum horizon:** The pre-intervention period must contain enough observations for differencing ($T-1$ useful observations) and for ElasticNet and CVXPY optimization to be numerically stable. In practice, $T \geq 30$ days is recommended.

---

## 3. Parameters

### 3.1 Required Parameters

| Parameter | Type | Description |
|:---|:---|:---|
| `filepath` | `str` | Absolute or relative path to the input CSV file (daily factor by geographies). |
| `date_col` | `str` | string/datetime column representing the date of each sequential observation. |

### 3.2 Optional Parameters (Experiment Configuration)

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `geos` | `list[str]` | `None` | Specifies or restricts the total range of markets to be read from the database. If `None`, uses all numeric columns in the CSV. |
| `n_treatment` | `int` | `3` | Number of geographies in the treatment group. Used if `fixed_treatment` is `None`. The software will generate combinatorial partitions containing this *N* number of places in each simulation. |
| `fixed_treatment` | `list[str]` | `None` | Named list of geographies that your team actively chose to intervene in. Disables the `n_treatment` combinatorial and analyzes the optimal scenario focused solely on safeguarding the data of the chosen targets. |
| `start_date` | `str` | `None` | Start date of the analysis period in `YYYY-MM-DD` format. Filters previous records, ensuring past synergies are tested **without influence** from the treatment itself (Data Leak). |
| `end_date` | `str` | `None` | End date of the analysis period in `YYYY-MM-DD` format. Filters subsequent records. |

### 3.3 Algorithmic Hyperparameters

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `use_elasticnet` | `bool` | `True` | Enables control pre-filtering via ElasticNet. When `False`, all controls enter CVXPY optimization directly. |
| `search_mode` | `str` | `"auto"` | Search strategy: `"exhaustive"`, `"ranking"`, or `"auto"`. See section 5.1 for full details. |
| `alpha` | `float` \| `list` | `0.01` | ElasticNet regularization intensity. Accepts a single value or a list (e.g., `[0.001, 0.01, 0.1]`) to perform automatic grid search. |
| `l1_ratio` | `float` \| `list` | `0.5` | Balance between L1 (Lasso) and L2 (Ridge). Accepts a single value or a list (e.g., `[0.2, 0.5, 0.8]`) for grid search. |
| `n_jobs` | `int` | `None` | Number of workers for parallel processing. If `None`, auto-detects the best value for the system. |
| `verbose` | `bool` | `True` | Displays progressive logs and progress bar (via `tqdm`). |
| `show_results` | `bool` | `True` | Prints the formatted results table at the end of execution. |

### 3.5 ElasticNet Configuration (Grid Search)

Unlike previous versions, grid search is now optional and controlled directly by the `alpha` and `l1_ratio` parameters. If lists are provided, the function will test all combinations and select the model that minimizes error and maximizes correlation.

| Hyperparameter | Suggested Default | Meaning |
|:---|:---|:---|
| `alpha` | `0.01` | Regularization intensity. Smaller values → less penalty → more controls survive. |
| `l1_ratio` | `0.5` | Balance between sparsity and smoothness. High values → more sparsity → fewer controls. |

This generates a $3 \times 3 = 9$ grid of configurations evaluated **for each treatment combination**. Only the best local result (lowest `ser`) is retained per combination.

---

## 4. Theoretical Basis and Hypotheses

### 4.1 Foundation: Synthetic Control Method (Abadie et al., 2003, 2010)

The function implements a variant of the Synthetic Control Method (SCM). SCM proposes that, in the absence of a randomized experiment, the counterfactual trajectory of a treated unit can be estimated as a **weighted convex combination** of untreated units:

$$
\hat{Y}_{T,t} = \sum_{j \in D} w_j \cdot Y_{j,t} \quad \text{with} \quad w_j \geq 0, \quad \sum_j w_j = 1
$$

where $\hat{Y}_{T,t}$ is the synthetic value at time $t$, $Y_{j,t}$ is the observed value of donor $j$, and $w_j$ is the optimized weight.

### 4.2 Assumed Hypotheses

1. **Latent Parallelism (Generalized Parallel Trends):** There exists a convex linear combination of controls that replicates the treatment trajectory in the pre-intervention period. If this combination does not exist, Synthetic Control is inherently unsuitable for the problem.

2. **Structural Stability:** The co-movement patterns observed in the pre-intervention period hold in the post-intervention period, except for the causal effect of the treatment. Asymmetric exogenous shocks (e.g., localized natural disaster) violate this hypothesis.

3. **Non-anticipation:** Treatment units do not change their behavior before the treatment start date. The function operates exclusively on pre-intervention data, but if there is temporal leakage, the historical fit will be artificially inflated.

4. **No interference (SUTVA):** The treatment of one geo does not affect the outcomes of control geos. In digital marketing contexts with geographic spillover, this hypothesis may be violated.

### 4.3 Implemented Extension: Filtering the "Crowd" with ElasticNet

Classic SCM (Abadie) does not perform variable selection — all units in the donor pool participate in optimization. With many cities in control, comparing all would generate too much stochastic noise (*overfitting*). The function introduces a prior **ElasticNet regularization** stage that simultaneously evaluates dozens of donors and naturally "zeros out" the relevance of cities that do not help approximate the Treatment:

$$
\min_{w} \frac{1}{2T} \lVert Xw - y \rVert_2^2 + \alpha \cdot \lambda_1 \lVert w \rVert_1 + \frac{\alpha}{2} (1 - \lambda_1) \lVert w \rVert_2^2
$$

The L1 penalty induces sparsity (zeros out coefficients of irrelevant controls), while L2 stabilizes the solution when controls are multicollinear.

**The Positive Rule:** If a candidate city receives a negative weight (i.e., it grows in isolation when the main target falls), the vector discards it. Controls require real vector similarity, not bizarre mirrored correlations.

*(Note: When triggered via the `use_elasticnet=False` parameter in purist DiD settings, the algorithm intervenes to create controls under strict fixed restriction purely based on holistic vector correlation [1/N], avoiding convex penalty).*

### 4.4 Simplifications

- **Mean normalization:** CVXPY optimization normalizes series by dividing by the temporal mean ($y / \bar{y}$, $X / \bar{X}$), instead of using Abadie's classic V-weighted normalization. This simplifies the problem but loses the ability to weight auxiliary covariates.
- **No intercept:** The synthetic model does not include an intercept ($\hat{Y} = Xw$, not $\hat{Y} = Xw + b$), forcing the control to explain both level and dynamics.
- **Discrete grid search:** ElasticNet hyperparameters are searched on a fixed 9-point grid, not via cross-validation. This is pragmatic (speed) but may not find the global optimization optimum.

---

## 5. Mechanism / Transformation

The function executes a 6-stage sequential *pipeline* for each treatment group candidate.

### 5.0 Data Preparation

```
Raw CSV → date parsing → temporal filtering → daily grouping → chronological ordering
```

The output is a DataFrame $\mathbf{D} \in \mathbb{R}^{T \times n}$ where $T$ is the number of days and $n$ is the number of geos.

### 5.1 Search Strategy Selection (`search_mode`)

Search behavior changes depending on how the function is called:

| Condition | Effective Mode | Behavior |
|:---|:---|:---|
| `fixed_treatment` provided | `fixed` | Evaluates each fixed geo individually, with cross-isolation among all fixed ones |
| `n_treatment == 1` | `exhaustive` | Each geo is tested as a unit treatment ($C(n,1) = n$ iterations) |
| `search_mode="exhaustive"` | `exhaustive` | Generates all $C(n, k)$ combinations and evaluates each as a group |
| `search_mode="ranking"` | `ranking` | Two-phase greedy heuristic (see 5.1.2) |
| `search_mode="auto"` | dynamic | $C(n,k) > 1000 \Rightarrow$ `ranking`; otherwise, `exhaustive` |

#### 5.1.1 Exhaustive Mode (`"exhaustive"`) — Full Combinatorial Search

Exhaustive search tests **all** possible mathematical combinations $C(n, k)$, where $n$ is the total number of geographies and $k$ is `n_treatment`:

$$
C(n, k) = \frac{n!}{k! \cdot (n - k)!}
$$

**Practical scale examples:**

| Geos ($n$) | Treatments ($k$) | Combinations $C(n,k)$ |
|:---:|:---:|---:|
| 10 | 3 | 120 |
| 15 | 3 | 455 |
| 20 | 5 | 15,504 |
| 25 | 5 | 53,130 |
| 27 | 5 | 80,730 |
| 30 | 5 | 142,506 |

**Algorithmic Flow:**

Each combination is evaluated and ranked by the **Synthetic Error Ratio (SER)** — a metric that penalizes high error and/or low correlation between treatment and its Synthetic Control. Formally, $\text{SER} = \sigma_e / (\rho + 10^{-6})$, where $\sigma_e$ is the standard deviation of the residual and $\rho$ is the Pearson correlation (see full details in section 6).

```
For EACH combination C(n,k):
  1. Define treatment group (e.g., [SP, RJ, MG, BA, PR])
  2. Define donor pool = all geos EXCEPT treatment
  3. Create target series: y = mean(treatments)
  4. Apply ElasticNet to filter irrelevant controls
  5. Optimize weights via CVXPY (Synthetic Control)
  6. Calculate metrics: std_residual, correlation, SER
  7. Store result
                    ↓
Order ALL combinations by ascending SER
                    ↓
Return TOP 5 as RECOMMENDATION 0..4
```

**Key characteristic:** each iteration evaluates the treatment group **as a composite unit** — the target series `y` is the mean of the `k` geos acting together. This captures **synergies and redundancies** between geos that an individual search would never detect.

**Result — 5 Independent Alternatives:**

Exhaustive mode returns the **top 5 complete combinations**, ranked by SER:

```
RECOMMENDATION 0 | Treatment: [Geo3, Geo8, Geo15, Geo21, Geo27] | SER: 0.312  ← best
RECOMMENDATION 1 | Treatment: [Geo3, Geo8, Geo15, Geo21, Geo24] | SER: 0.318
RECOMMENDATION 2 | Treatment: [Geo2, Geo8, Geo15, Geo21, Geo27] | SER: 0.325
RECOMMENDATION 3 | Treatment: [Geo3, Geo9, Geo15, Geo21, Geo27] | SER: 0.331
RECOMMENDATION 4 | Treatment: [Geo4, Geo8, Geo15, Geo21, Geo27] | SER: 0.337
```

If RECOMMENDATION 0 is not feasible due to operational constraints (e.g., a city unavailable for the campaign), the user can adopt RECOMMENDATION 1 with confidence — each alternative was **independently optimized** over the full space.

---

#### 5.1.2 Ranking Mode (`"ranking"`) — Two-Phase Greedy Search

Ranking search is a **greedy heuristic** that drastically reduces the number of evaluations by assuming the best individual geos form the best group.

**Algorithmic Flow:**

```
PHASE 1 — Individual Screening (n evaluations):
  For EACH geo in isolation:
    1. Treat geo as unit treatment
    2. Donor pool = all other geos
    3. Fit Synthetic Control
    4. Calculate individual SER
                    ↓
  Order n geos by ascending SER
  Select top-k (best individuals)

PHASE 2 — Re-evaluation with Cross-Exclusion (k evaluations):
  For EACH geo in top-k:
    1. Treat geo as treatment
    2. Donor pool = all geos EXCEPT all top-k
    3. Re-fit Synthetic Control with full isolation
    4. Re-calculate SER
                    ↓
Return k clusters as TEST CLUSTER 0..k-1
```

**Total evaluations:** $n + k$ (e.g., 27 + 5 = 32), versus 80,730 in exhaustive.

**Result — A Single Group:**

Ranking returns **a single treatment group** composed of the top-k individual geos. There are no alternatives — if the group doesn't work, manual reconfiguration of parameters is required.

> [!WARNING]
> **Compositional Fallacy of Greedy Search**
> Ranking assumes the best individual geos form the best group. This is not always true. Two geos might be individually excellent but highly correlated with each other — when combined, they "compete" for the same controls and degrade the quality of the joint Synthetic Control. Exhaustive mode is immune to this problem because it evaluates each combination as a full group.

---

#### 5.1.3 Auto Mode (`"auto"`) — Adaptive Selection (default)

Auto mode decides between exhaustive and ranking based on problem cardinality:

```python
if C(n, k) > 2000:
    use "ranking"
else:
    use "exhaustive"
```

This ensures small problems (up to ~1,000 combinations) receive the full optimal solution, while large problems avoid prohibitive execution times.

---

#### Search Mode Comparative Table

| Feature | `"exhaustive"` | `"ranking"` | `"auto"` |
|:---|:---|:---|:---|
| **What each iteration tests** | Full group of $k$ geos | 1 isolated geo | Depends on $C(n,k)$ |
| **No. of evaluations** | $C(n, k)$ | $n + k$ | Adaptive |
| **Captures synergy between geos** | Yes | No | Depends |
| **Alternatives returned** | Top 5 Recommendations | Single group | Depends |
| **Speed** | Slow for large $C(n,k)$ | Fast | Adaptive |
| **When to use** | $C(n,k) \leq 10,000$ or maximum precision | Bases with many geos ($n > 30$) | General use |

> [!TIP]
> **Practical Recommendation:** For high-financial-impact decisions, force `search_mode="exhaustive"` even with longer execution times. The 5 alternatives returned offer operational flexibility that greedy search does not provide. Reserve `"ranking"` for rapid prototyping or bases with dozens of geos where exhaustive would be computationally infeasible.

### 5.2 log-diff Transformation (`log_diff_transform`)

To prevent the model from being misled by long-term trends or scale distortions, a log and difference transformation is applied. For each geographic series $Y_{j,t}$:

$$
Z_{j,t} = \Delta \log Y_{j,t} = \log Y_{j,t} - \log Y_{j,t-1} = \log\left(\frac{Y_{j,t}}{Y_{j,t-1}}\right)
$$

**Triple Purpose:**

1. **Stationarization:** Removes deterministic trends. The resulting series represents relative growth rates, not absolute levels.
2. **Scale normalization:** Series with very different magnitudes (e.g., São Paulo vs. Macapá) become comparable on a logarithmic scale.
3. **Variance stabilization:** The logarithm compresses the right tail, reducing heteroscedasticity typical of revenue/sales series.

**Consequence:** Subsequent optimization operates on **daily log returns**, not raw values. This focuses the model on aligning *relative dynamics* (day-to-day under/over-performance patterns), making the series stationary.

The operation consumes 1 observation (the first difference eliminates $t=0$), reducing the series from $T$ to $T-1$ points.

### 5.3 Control Pre-filtering (ElasticNet)

With `use_elasticnet=True`, each combination undergoes the following procedure:

1. **Standardization:** The controls matrix $X \in \mathbb{R}^{(T-1) \times (n-k)}$ is standardized (zero mean, unit variance) via `StandardScaler`.

2. **Grid Search:** For each $(\alpha, \lambda_1)$ pair in the grid $\{0.001, 0.01, 0.1\} \times \{0.2, 0.5, 0.8\}$, the following is fitted:

$$
\hat{w} = \arg\min_{w} \frac{1}{2(T-1)} \lVert Xw - y \rVert_2^2 + \alpha \lambda_1 \lVert w \rVert_1 + \frac{\alpha(1-\lambda_1)}{2} \lVert w \rVert_2^2
$$

3. **Positive Selection:** Only controls with $\hat{w}_j > 0$ are selected. If all coefficients are $\leq 0$, the control with the largest $\lvert \hat{w}_j \rvert$ is forced as selected (safety fallback).

**Intuition:** ElasticNet acts as a "relevance filter" that separates the crowd of candidates. Controls that do not contribute to explaining treatment variation receive zero weight (L1/Lasso contribution). Controls with a negative relationship (moving in the opposite direction of treatment) are explicitly excluded by the positive coefficient rule.

### 5.4 Synthetic Control Optimization (CVXPY)

Controls that survive the ElasticNet filter undergo convex optimization to define definitive weights. This stage operates on the **original data** (not transformed), ensuring weights reflect real-scale fit.

**Prior Normalization:**

$$
\tilde{y}_t = \frac{y_t}{\bar{y}}, \qquad \tilde{X}_{j,t} = \frac{X_{j,t}}{\bar{X}_j}
$$

where $\bar{y} = \frac{1}{T}\sum_t y_t$ and $\bar{X}_j = \frac{1}{T}\sum_t X_{j,t}$

**Optimization Problem:**

$$
\min_{w} \sum_{t=1}^{T} \left(\tilde{y}_t - \sum_{j} w_j \tilde{X}_{j,t}\right)^2
$$

subject to:

$$
w_j \geq 0 \quad \forall j, \qquad \sum_j w_j = 1
$$

Resolved via the SCS (Splitting Conic Solver) from the `cvxpy` package.

**Final Pruning:** Controls with $w_j < 0.001$ (0.1%) are removed. Remaining weights are re-normalized to sum to 1:

$$
w_j^* = \frac{w_j}{\sum_{i} w_i} \quad \text{for } w_j \geq 0.001
$$

If a city receives an insignificant weight (below 0.1%), it is pruned from the definitive *donor pool*.

### 5.5 Fit Evaluation (Metrics)

With final weights $w^*$ and pruned controls, the predicted series is calculated in **log-diff space** (transformed):

$$
\hat{y}_t = \sum_j w_j^* Z_{j,t} \quad \text{(without ElasticNet)}
$$

or

$$
\hat{y}_t = f_{\text{ElasticNet}}(X_{\text{scaled}}) \quad \text{(with ElasticNet, refit on selected controls)}
$$

The residual is $e_t = y_t - \hat{y}_t$, and computed metrics are:

| Metric | Formula | Interpretation |
|:---|:---|:---|
| `std_residual` | $\sigma_e = \sqrt{\frac{1}{T-1}\sum_t (e_t - \bar{e})^2}$ | Daily error dispersion. Low = stable synthetic. |
| `rmspe` | $\text{RMSPE} = \sqrt{\frac{1}{T-1}\sum_t e_t^2}$ | Absolute error magnitude. Includes bias. |
| `correlation` | $\rho = \text{corr}(y, \hat{y})$ | Directional sync. High = in-phase movements. |
| `synthetic_error_ratio` | $\text{SER} = \frac{\sigma_e}{\rho + 10^{-6}}$ | Final ranking metric. See section 6. |

### 5.6 Local Best Hyperparameter Selection

Within the 9 ElasticNet grid configurations, only the candidate with the **lowest `ser`** is retained to represent that treatment combination. This means each geo combination contributes exactly one result to the global ranking.

---

## 6. Objective Function: Synthetic Error Ratio (SER)

### 6.1 Ranking Metric

The final algorithm evaluates predicted vs. actual oscillations with the winning group and ranks all options based on two fundamental variables:

- **Standard Residual (`std_residual`)**: Measures absolute historical daily error between actual and simulated. Represents error band width — "wider" means less precise partition (low preferred).
- **Correlation (`correlation`)**: Measures synchrony level of "peaks and valleys" in prediction lines (high preferred).

To resolve classic Causal Inference tie-breakers (e.g., highly correlated models but with high absolute error levels that widen Geo-Test confidence intervals), the function creates a penalizing heuristic:

$$
\text{SER} = \frac{\sigma_e}{\rho + \epsilon}, \quad \epsilon = 10^{-6}
$$

The algorithm divides error band width by correlation, forcing ranking to severely prioritize models whose curves have lower historical standard deviation, *provided* they move together. This ensures "Cluster 0" will represent the most historically static and methodologically secure Synthetic Control.

### 6.2 Why not use RMSPE alone?

RMSPE, standard literature metric (Abadie et al., 2010), measures absolute prediction error. However, in corporate contexts with high volatility:

- Low RMSPE might come from a "flat" control whose mean accidentally nears treatment mean without tracking seasonal dynamics.
- These **"Zombie Controls"** pass classic RMSPE criteria but collapse out-of-sample when exogenous shocks hit treatment and synthetic doesn't react.

The $\rho$ denominator penalizes this: a flat control will have $\rho \approx 0$, inflating SER to $+\infty$ and eliminating it from ranking.

### 6.3 Why not use correlation alone?

Correlation measures directional synchrony but is scale-invariant. Two signals with $\rho = 0.99$ can have residuals of completely different magnitudes. Without $\sigma_e$ in the numerator, ranking would prioritize "synchronized from afar" models — correlated but with wide error bands making Geo-Test statistically weak (wide post-treatment confidence intervals).

> [!NOTE]
> **Academic Note on Operational Heuristic**
> In classic statistics (Abadie et al., 2010), Synthetic Control optimization aims solely to minimize Mean Squared Prediction Error (RMSPE) in the pre-treatment period. However, in Applied Causality for business and marketing, absolute RMSPE minimization suffers from *variance suppression*, where modeling might select "fixed" or "flat" matrices with low passive historical error that don't react equivalent to seasonal exogenous shocks.
>
> **Synthetic Error Ratio** doesn't propose to be a new fundamental theorem, but acts as a purely pragmatic **Regularizing Loss Function**. Assuming Correlation as "Signal" (latent structural behavior) and Residual as "Noise" (useless stochastic variance), the mathematical division simulates inverse maximization of *Signal-to-Noise Ratio (SNR)*. This empirically forces exclusion of "Zombie Controls" and ensures Out-of-Sample Robustness against modern corporate instability.

### 6.4 Optimization Behavior

The function **does not optimize SER directly** via gradient or solver. SER is a *post-hoc* evaluation metric: each configuration (combination + hyperparameters) is evaluated independently and SER is calculated at the end. Ranking is done by simple sorting of SER vectors.

Formally, the returned solution is:

$$
S^* = \arg\min_{S \in \mathcal{C}} \text{SER}(S)
$$

where minimization is by enumeration (exhaustive) or greedy heuristic (ranking).

---

## 7. Outputs

### 7.1 Type

`list[dict]` — List of dictionaries, each representing an evaluated treatment/control partition.

### 7.2 Return Behavior by Mode

| Mode | Quantity Returned | Label in Output |
|:---|:---|:---|
| `exhaustive` | Top 5 (of $C(n,k)$ evaluated) | `RECOMMENDATION 0..4` |
| `ranking` | $k$ clusters (one per treatment geo) | `TEST CLUSTER 0..k-1` |
| `fixed` | $|F|$ clusters (one per fixed geo) | `TEST CLUSTER 0..|F|-1` |

### 7.3 Dictionary Schema

```python
{
    "treatment":            list[str],    # Geos in treatment group
    "control":              list[str],    # Geos in final donor pool (post-pruning)
    "control_weights":      list[float],  # Weights ∈ (0,1], Σ = 1.0
    "correlation":          float,        # ρ ∈ [-1, 1] (Pearson, log-diff space)
    "std_residual":         float,        # σ_e ≥ 0 (log-diff space)
    "rmspe":                float,        # RMSPE ≥ 0 (log-diff space)
    "ser":                  float,        # SER = σ_e / (ρ + 1e-6)
    "n_controls":           int,          # No. of controls pre-CVXPY pruning
    "alpha":                float,        # Best ElasticNet α
    "l1_ratio":             float         # Best ElasticNet λ₁
}
```

### 7.4 Ordering

The list is always returned in **ascending** order of SER (when `use_elasticnet=True`) or RMSPE (when `use_elasticnet=False`). Index `0` is always the best experimental design found.

---

## 8. Result Interpretation

### 8.1 Reading RECOMMENDATION 0

Consider the following hypothetical result:

```
RECOMMENDATION 0
  treatment:            [curitiba, florianopolis, porto_alegre]
  control:              [belo_horizonte, goiania, brasilia, recife]
  control_weights:      [0.42, 0.31, 0.18, 0.09]
  correlation:          0.9213
  std_residual:         0.0387
  synthetic_error_ratio: 0.0420
```

**Line-by-line interpretation:**

- **`treatment`**: These 3 cities should receive intervention (campaign, holdout, etc.). The model target series is the daily arithmetic mean of these 3 cities.
- **`control`**: These 4 cities compose the Synthetic Control. They **must not receive treatment** during experiment.
- **`control_weights`**: Belo Horizonte contributes 42% of synthetic, Goiânia 31%, Brasília 18%, and Recife 9%. Counterfactual will be: $\hat{Y}_t = 0.42 \cdot \text{BH}_t + 0.31 \cdot \text{GO}_t + 0.18 \cdot \text{BSB}_t + 0.09 \cdot \text{REC}_t$.
- **`correlation = 0.9213`**: In pre-intervention, daily variations (log-diff) of treatment and synthetic move in same direction on 92% of days. This sync level is high and indicative of good structural fit.
- **`std_residual = 0.0387`**: Daily error dispersion (log-diff scale) is ~3.9%. In absence of treatment, synthetic prediction is expected to oscillate ±3.9% around actual on typical day.
- **`synthetic_error_ratio = 0.0420`**: Consolidated metric. Values below 0.1 generally indicate excellent fit. Values above 0.5 suggest low-quality partition.

### 8.2 When to worry

| Signal | Diagnosis | Action |
|:---|:---|:---|
| SER > 0.3 | Weak fit — synthetic doesn't replicate treatment | Increase history, remove noisy geos, or reduce `n_treatment` |
| $\rho < 0.7$ | Structural sync loss | Investigate geo heterogeneity; consider regional clusters |
| $\sigma_e > 0.1$ | Wide residual — wide confidence interval | Accept lower power or seek geos with better co-movement |
| Weight > 0.8 in single control | Excessive dependency | Risk that idiosyncratic shock in this control invalidates full synthetic |

---

## 9. Limitations

### 9.1 Mathematical Limitations

1. **Grid local optimality:** ElasticNet grid search uses discrete 9-point grid. Optimal combination $(\alpha^*, \lambda_1^*)$ might reside outside, especially for atypical data.

2. **Evaluation duality:** ElasticNet is fit in log-diff space (stationary), but CVXPY weights optimized in original space (level). This shift between stages 5.3 and 5.4 might introduce inconsistency: relevant control in log-diff might fit poorly in level, and vice versa.

3. **No temporal cross-validation:** Evaluation uses full pre-intervention period for both fit and evaluation (in-sample). No temporal train/validation split, potentially leading to *overfitting* in short periods.

### 9.2 Computational Limitations

1. **Combinatorial explosion:** $C(50, 10) \approx 10^{10}$. For bases with $n > 30, k > 5$, exhaustive mode is computationally infeasible. `ranking` mode is a pragmatic solution but sacrifices global optimality guarantee.

2. **Cost per iteration:** Each evaluation involves 9 ElasticNet fits + 1 CVXPY optimization (SCS solver). For $C(n,k) = 80,730$, this results in ~726,570 model fits.

### 9.3 Statistical Limitations

1. **No multiple comparisons correction:** Ranking $C(n,k)$ combinations by SER selects best *ex-post* fit. With many geos and few observations, "best" might be a selection artifact (analogous to *p-hacking*).

2. **Violable stability assumption:** Function doesn't formally test if co-movement patterns are stable over time. $\rho = 0.95$ over 90 days might mask structural break at 60 days compromising future validity.

3. **SER as heuristic:** SER lacks axiomatic foundation (not derived from probabilistic model). Motivated by Signal-to-Noise analogy, but no formal proof minimizing SER minimizes causal effect estimate bias.

### 9.4 Practical Limitations

1. **Zero values:** Frequent zero series (missing data, holidays) generate `log(0) = -∞`, corrupting pipeline. Requires prior treatment (drop, imputation, or $\log(x + 1)$ — the latter **not** implemented internally).

2. **Short-series geos:** If `start_date`/`end_date` limit horizon to < 15 days, differencing + grid search might operate with $T < 14$ observations, insufficient for ElasticNet numerical stability.

---

## 10. Use Cases

### 10.1 When to use

| Scenario | Recommended Configuration |
|:---|:---|
| Geo-Test design before digital campaign | `n_treatment=3..5`, `search_mode="exhaustive"` (if feasible) |
| Validation of team-defined holdout | `fixed_treatment=["geo1", "geo2"]` |
| Rapid prototyping with many geos (>30) | `search_mode="ranking"` |
| Sensitivity analysis of treatment selection | Compare `clusters[0]` to `clusters[4]` in exhaustive mode |
| Pure DiD experiments (no regularization) | `use_elasticnet=False` |

### 10.2 When NOT to use

| Scenario | Reason | Alternative |
|:---|:---|:---|
| High zero data (binary sales) | $\log$ transformation fails | Pre-process with $\log(x+1)$ or use classic DiD |
| Single geo (n=1 treatment, n=1 control) | No degrees of freedom to optimize | Use Bayesian Structural Time Series (CausalImpact) |
| < 15 pre-treatment observations | Numerical instability | Collect more data or use non-parametric method |
| Strong geo spillover | SUTVA violation | Use geo buffers or cluster randomization |
| Constant level treatment effects | log-diff evaluation; level effects might be diluted | Consider level model with covariates |

---

## 11. Simple Example

### 11.1 Setup

Consider `sales.csv` with 60 days of sales for 6 cities:

| date | SP | RJ | BH | CWB | POA | REC |
|:---|---:|---:|---:|---:|---:|---:|
| 2026-01-01 | 1000 | 800 | 400 | 300 | 250 | 200 |
| 2026-01-02 | 1020 | 810 | 410 | 305 | 255 | 198 |
| ... | ... | ... | ... | ... | ... | ... |
| 2026-03-01 | 1150 | 870 | 430 | 320 | 270 | 210 |

### 11.2 Call

```python
from reallift.geo.discovery import discover_geo_clusters

# Exploratory Example: Exhaustive search for best 2-treatment design.
clusters = discover_geo_clusters(
    filepath="sales.csv",
    date_col="date",
    n_treatment=2,
    search_mode="exhaustive",
    n_jobs=None,
    verbose=True
)
```

### 11.3 Internal Process

1. **Combinations generated:** $C(6, 2) = 15$

2. **Per combination** (e.g., `(SP, RJ)`):
   - Treatment: $y_t = \frac{\text{SP}_t + \text{RJ}_t}{2}$
   - Donor pool: `[BH, CWB, POA, REC]`
   - `log_diff_transform` → 59 daily return observations
   - Run 9 ElasticNet configs → filter controls → CVXPY optimize weights
   - Calculate metrics → retain best of 9 configs

3. **Total evaluations:** $15 \times 9 = 135$ model fits

4. **Ordering:** 15 combinations ordered by SER

### 11.4 Output (hypothetical)

```python
clusters[0]
# {
#     "treatment": ["CWB", "POA"],
#     "control": ["BH", "REC"],
#     "control_weights": [0.72, 0.28],
#     "correlation": 0.9450,
#     "std_residual": 0.0310,
#     "ser": 0.0328,
#     "n_controls": 2,
#     "alpha": 0.01,
#     "l1_ratio": 0.5
# }
```

### 11.5 Interpretation

The algorithm identified Curitiba + Porto Alegre, when treated together, are best replicated by synthetic composed of 72% Belo Horizonte + 28% Recife. This was the best design of 15 possible, with SER = 0.0328.

If Curitiba can't be treated (operational constraint), adopt `clusters[1]` confidently — this alternative was independently evaluated over the full space.

### 11.6 Fixed Treatment Example

```python
# Directed Example: Forcing control groups for Active Campaign units.
best_clusters = discover_geo_clusters(
    filepath="historical_sales.csv",
    date_col="billing_date",
    fixed_treatment=["sao_paulo", "rio_de_janeiro"],
    start_date="2026-01-01"
)

for sub_test in best_clusters:
     print(f"Target: {sub_test['treatment'][0]} => Explained by: {sub_test['control']}")
```

---

## References

- Abadie, A., & Gardeazabal, J. (2003). *The Economic Costs of Conflict: A Case Study of the Basque Country.* American Economic Review, 93(1), 113-132.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). *Synthetic Control Methods for Comparative Case Studies.* Journal of the American Statistical Association, 105(490), 493-505.
- Zou, H., & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net.* Journal of the Royal Statistical Society: Series B, 67(2), 301-320.
- O'Donoghue, B., Chu, E., Parikh, N., & Boyd, S. (2016). *Conic Optimization via Operator Splitting and Homogeneous Self-Dual Embedding.* Journal of Optimization Theory and Applications, 169(3), 1042-1068.
