import numpy as np
from ..config.defaults import DEFAULT_N_BOOT

def bootstrap_significance(effect, post_synth, n_boot=DEFAULT_N_BOOT, random_state=None) -> dict:
    """
    Perform bootstrap significance testing.

    Parameters:
        effect (np.ndarray): Effect array.
        post_synth (np.ndarray): Post-treatment synthetic values.
        n_boot (int): Number of bootstrap samples.
        random_state (int or np.random.Generator): Random state for reproducibility.

    Returns:
        dict: Bootstrap results.
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

    for _ in range(n_boot):
        idx = rng.choice(len(effect), size=len(effect), replace=True)
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

    ci_lower_abs = np.percentile(boot_means_abs, 2.5)
    ci_upper_abs = np.percentile(boot_means_abs, 97.5)
    ci_lower_pct = np.percentile(boot_means_pct, 2.5)
    ci_upper_pct = np.percentile(boot_means_pct, 97.5)
    
    ci_lower_total_abs = np.percentile(boot_totals_abs, 2.5)
    ci_upper_total_abs = np.percentile(boot_totals_abs, 97.5)
    ci_lower_total_pct = np.percentile(boot_totals_pct, 2.5)
    ci_upper_total_pct = np.percentile(boot_totals_pct, 97.5)

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