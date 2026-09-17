"""Transfer Predictor: learn to predict functional forward transfer T_r from
boundary-available features.

This package is intentionally lazy-imported so standard CL and meta-learning
modules can be imported without pulling in the heavier scikit-learn stack
unless the transfer-predictor workflow is explicitly requested.
"""

__all__ = [
    "build_dataset",
    "FeatureProvenance",
    "build_feature_matrix",
    "get_boundary_features",
    "LOTOConfig",
    "run_loto_cv",
    "get_model",
    "get_all_model_names",
    "compute_metrics",
    "aggregate_fold_results",
    "print_summary",
    "save_results",
]


def __getattr__(name: str):
    if name == "build_dataset":
        from .data_collection import build_dataset
        return build_dataset
    if name in {"FeatureProvenance", "build_feature_matrix", "get_boundary_features"}:
        from .features import FeatureProvenance, build_feature_matrix, get_boundary_features
        mapping = {
            "FeatureProvenance": FeatureProvenance,
            "build_feature_matrix": build_feature_matrix,
            "get_boundary_features": get_boundary_features,
        }
        return mapping[name]
    if name in {"LOTOConfig", "run_loto_cv"}:
        from .loto_validation import LOTOConfig, run_loto_cv
        mapping = {"LOTOConfig": LOTOConfig, "run_loto_cv": run_loto_cv}
        return mapping[name]
    if name in {"get_model", "get_all_model_names"}:
        from .models import get_model, get_all_model_names
        mapping = {"get_model": get_model, "get_all_model_names": get_all_model_names}
        return mapping[name]
    if name in {"compute_metrics", "aggregate_fold_results", "print_summary", "save_results"}:
        from .evaluate import (
            aggregate_fold_results,
            compute_metrics,
            print_summary,
            save_results,
        )
        mapping = {
            "compute_metrics": compute_metrics,
            "aggregate_fold_results": aggregate_fold_results,
            "print_summary": print_summary,
            "save_results": save_results,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")