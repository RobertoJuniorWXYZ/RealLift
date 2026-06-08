import numpy as np
from ..config.defaults import DEFAULT_N_BOOT

def bootstrap_significance(effect, post_synth, n_boot=DEFAULT_N_BOOT, conf_level=0.95, random_state=None) -> dict:
    """
    Perform significance testing using Circular Moving Block Bootstrap (CMBB).

    CMBB treats the series as circular so every observation participates in exactly
    ``block_size`` blocks, eliminating the end-of-series truncation bias that inflates
    confidence intervals in the standard (linear) MBB.

    Block size defaults to 7 days (weekly seasonality) for series >= 14 observations,
    3 days for series >= 6, and falls back to i.i.d. bootstrap for very short series.

    Parameters:
        effect (np.ndarray): Observed effect array (Observed - Synthetic).
        post_synth (np.ndarray): Post-treatment synthetic values array.
        n_boot (int): Number of bootstrap iterations.
        conf_level (float): Confidence level for intervals (default: 0.95).
        random_state (int or np.random.Generator): Seed or generator for reproducibility.

    Returns:
        dict: Absolute and percentage confidence intervals, p-values, and raw bootstrap arrays.
    """
    if random_state is not None:
        if isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
    else:
        rng = np.random.default_rng()

    boot_means_abs = []
    boot_means_pct = []
    boot_totals_abs = []
    boot_totals_pct = []
    
    # Handle division by zero if post_synth contains zeros
    post_synth_safe = np.where(post_synth == 0, 1e-10, post_synth)
    effect_pct = effect / post_synth_safe

    n = len(effect)
    
    # Moving Block Bootstrap to account for autocorrelation
    # Default to 7-day blocks (weekly seasonality) if data is long enough
    block_size = 7 if n >= 14 else (3 if n >= 6 else 1)
    
    if block_size > 1:
        # CMBB: all n positions are valid starts; wrap-around via modulo.
        possible_starts = np.arange(n)
        blocks_needed = int(np.ceil(n / block_size))

        for _ in range(n_boot):
            start_indices = rng.choice(possible_starts, size=blocks_needed, replace=True)
            idx = []
            for start_idx in start_indices:
                idx.extend((start_idx + j) % n for j in range(block_size))
            idx = np.array(idx[:n])  # truncate to exactly n
            
            sample_abs = effect[idx]
            sample_pct = effect_pct[idx]
            sample_synth = post_synth[idx]
            
            boot_means_abs.append(sample_abs.mean())
            boot_means_pct.append(sample_pct.mean())
            boot_totals_abs.append(sample_abs.sum())
            
            sum_synth = sample_synth.sum()
            sum_synth_safe = 1e-10 if sum_synth == 0 else sum_synth
            boot_totals_pct.append(sample_abs.sum() / sum_synth_safe)
    else:
        # Fallback to standard i.i.d bootstrap for very short series
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            sample_abs = effect[idx]
            sample_pct = effect_pct[idx]
            sample_synth = post_synth[idx]
            
            boot_means_abs.append(sample_abs.mean())
            boot_means_pct.append(sample_pct.mean())
            boot_totals_abs.append(sample_abs.sum())
            
            sum_synth = sample_synth.sum()
            sum_synth_safe = 1e-10 if sum_synth == 0 else sum_synth
            boot_totals_pct.append(sample_abs.sum() / sum_synth_safe)

    boot_means_abs = np.array(boot_means_abs)
    boot_means_pct = np.array(boot_means_pct)
    boot_totals_abs = np.array(boot_totals_abs)
    boot_totals_pct = np.array(boot_totals_pct)

    alpha = 1.0 - conf_level
    lower_p = (alpha / 2.0) * 100
    upper_p = (1.0 - alpha / 2.0) * 100

    ci_lower_abs = np.percentile(boot_means_abs, lower_p)
    ci_upper_abs = np.percentile(boot_means_abs, upper_p)
    ci_lower_pct = np.percentile(boot_means_pct, lower_p)
    ci_upper_pct = np.percentile(boot_means_pct, upper_p)
    
    ci_lower_total_abs = np.percentile(boot_totals_abs, lower_p)
    ci_upper_total_abs = np.percentile(boot_totals_abs, upper_p)
    ci_lower_total_pct = np.percentile(boot_totals_pct, lower_p)
    ci_upper_total_pct = np.percentile(boot_totals_pct, upper_p)

    p_value_boot = 2 * min(
        np.mean(boot_means_abs <= 0),
        np.mean(boot_means_abs >= 0)
    )

    return {
        "ci_lower_abs": ci_lower_abs,
        "ci_upper_abs": ci_upper_abs,
        "ci_lower_pct": ci_lower_pct,
        "ci_upper_pct": ci_upper_pct,
        "ci_lower_total_abs": ci_lower_total_abs,
        "ci_upper_total_abs": ci_upper_total_abs,
        "ci_lower_total_pct": ci_lower_total_pct,
        "ci_upper_total_pct": ci_upper_total_pct,
        "p_value_boot": p_value_boot,
        "boot_means_abs": boot_means_abs,
        "boot_means_pct": boot_means_pct,
        "boot_totals_abs": boot_totals_abs,
        "boot_totals_pct": boot_totals_pct
    }