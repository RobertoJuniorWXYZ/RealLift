# Type schemas for standardized outputs

from typing import Dict, List, Any

# Schema for split results
SplitResult = Dict[str, Any]  # e.g., {"treatment": [...], "control": [...], "std_residual": float, ...}

# Schema for duration results
DurationResult = Dict[str, Any]  # e.g., {"summary": {...}, "power_curve": pd.DataFrame}

# Schema for validation results
ValidationResult = Dict[str, Any]  # e.g., {"summary": pd.DataFrame, "outputs": List[pd.DataFrame]}

# Schema for synthetic control results
SyntheticResult = Dict[str, Any]  # e.g., {"weights": {...}, "lift_mean_abs": float, ...}

# Schema for bootstrap results
BootstrapResult = Dict[str, Any]  # e.g., {"ci_lower": float, "ci_upper": float, "p_value": float}

# Schema for placebo results
PlaceboResult = Dict[str, Any]  # e.g., {"placebo_lifts": List[float], "p_value": float}

# Overall experiment result
ExperimentResult = Dict[str, Any]  # Consolidated output