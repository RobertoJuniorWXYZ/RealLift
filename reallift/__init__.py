# RealLift - Causal Inference Library for Lift Measurement

from .geo import (
    discover_geo_clusters,
    estimate_duration,
    validate_geo_clusters,
    run_synthetic_control,
    bootstrap_significance,
    run_placebo_tests
)
from .pipelines.geo_pipeline import run_geo_experiment, design_of_experiments
from .simulation import generate_geo_data, generate_simulated_intervention

__version__ = "0.3.0"