# RealLift - Causal Inference Library for Lift Measurement

from .geo import (
    find_best_geo_split,
    build_geo_clusters,
    estimate_duration,
    validate_geo_groups,
    run_synthetic_control,
    bootstrap_significance,
    run_placebo_tests
)
from .pipelines import run_geo_experiment
from .simulation import generate_geolift_data

__version__ = "0.1.0"