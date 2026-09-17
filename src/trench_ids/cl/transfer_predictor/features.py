"""Feature engineering for transfer predictor.

Constructs feature vectors from boundary-available information for predicting T_r.

Feature Provenance Categories:
- BOUNDARY: Available at task boundary before training t+1
- POST_HOC: Requires training on t+1 or future information (for ablation only)
- HISTORICAL: Derived from past task boundaries (available at boundary)

IMPORTANT: Only BOUNDARY and HISTORICAL features should be used for deployable predictors.
POST_HOC features are included for ablation/analysis only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from trench_ids.constants import FLOW_RELATIONS


class FeatureProvenance(Enum):
    BOUNDARY = "boundary"       # Available at t→t+1 boundary
    HISTORICAL = "historical"   # Derived from past boundaries
    POST_HOC = "post_hoc"       # Requires future info (ablation only)


@dataclass
class FeatureSpec:
    name: str
    provenance: FeatureProvenance
    description: str


# Feature specifications with provenance
FEATURE_SPECS = [
    # Core predictors (some boundary, some post-hoc)
    FeatureSpec("S_r", FeatureProvenance.POST_HOC, "Cosine similarity between new/old class means (post-hoc: uses trained encoder on new task)"),
    FeatureSpec("D_r", FeatureProvenance.HISTORICAL, "Representation drift at previous boundary"),
    FeatureSpec("G_r", FeatureProvenance.BOUNDARY, "Gradient alignment between old/new task objectives (can be computed at boundary)"),
    
    # Task-level features (boundary-available)
    FeatureSpec("n_classes_old", FeatureProvenance.BOUNDARY, "Number of classes in previous task(s)"),
    FeatureSpec("n_classes_new", FeatureProvenance.BOUNDARY, "Number of classes in upcoming task"),
    FeatureSpec("class_imbalance", FeatureProvenance.BOUNDARY, "Ratio of new/old class counts"),
    
    # Relation-level features (boundary-available)
    FeatureSpec("rel_param_count", FeatureProvenance.BOUNDARY, "Number of parameters in relation-specific modules"),
    FeatureSpec("rel_layer_count", FeatureProvenance.BOUNDARY, "Number of GNN layers with this relation"),
    
    # Historical transfer features (historical)
    FeatureSpec("hist_mean_Tr", FeatureProvenance.HISTORICAL, "Mean T_r for this relation at previous boundaries"),
    FeatureSpec("hist_std_Tr", FeatureProvenance.HISTORICAL, "Std of T_r for this relation at previous boundaries"),
    FeatureSpec("hist_positive_rate", FeatureProvenance.HISTORICAL, "Fraction of previous boundaries where T_r > 0 for this relation"),
    
    # Transition-level features
    FeatureSpec("transition_idx", FeatureProvenance.BOUNDARY, "Task transition index (1-5)"),
    FeatureSpec("is_early_transition", FeatureProvenance.BOUNDARY, "Whether transition is in first half (T1-T3)"),
    
    # Interaction features (boundary-available)
    FeatureSpec("S_r_x_D_r", FeatureProvenance.POST_HOC, "S_r * D_r interaction"),
    FeatureSpec("S_r_x_G_r", FeatureProvenance.POST_HOC, "S_r * G_r interaction"),
    FeatureSpec("D_r_x_G_r", FeatureProvenance.HISTORICAL, "D_r * G_r interaction"),
]


def get_feature_provenance(feature_name: str) -> FeatureProvenance:
    """Get provenance for a feature by name."""
    for spec in FEATURE_SPECS:
        if spec.name == feature_name:
            return spec.provenance
    return FeatureProvenance.POST_HOC  # Default to post-hoc for unknown


def get_boundary_features() -> list[str]:
    """Get list of feature names available at task boundary."""
    return [spec.name for spec in FEATURE_SPECS 
            if spec.provenance in (FeatureProvenance.BOUNDARY, FeatureProvenance.HISTORICAL)]


def get_all_feature_names() -> list[str]:
    """Get all feature names."""
    return [spec.name for spec in FEATURE_SPECS]


# Task class counts per task (from task design)
TASK_CLASSES = {
    1: ["Scanning"],
    2: ["Reconnaissance"],
    3: ["DDoS", "Infiltration"],
    4: ["DoS", "Injection"],
    5: ["Password", "Bot"],
    6: ["XSS", "BruteForce"],
}


# Relation parameter counts (approximate, from 3-layer GNN architecture)
RELATION_PARAM_COUNTS = {
    "originates": 12288,
    "terminated_by": 12288,
    "targeted_by": 12288,
    "protocol_of": 12288,
    "service_of": 12288,
}


def compute_historical_features(
    df: pd.DataFrame,
    transition: str,
    relation: str,
) -> dict[str, float]:
    """Compute historical features from previous transitions for this relation."""
    task = int(transition.split("_")[0][1:])
    prev_transitions = [f"t{t}_t{t+1}" for t in range(1, task)]
    
    prev_data = df[
        (df["transition"].isin(prev_transitions)) & 
        (df["relation"] == relation)
    ]["T_r"]
    
    if len(prev_data) == 0:
        return {
            "hist_mean_Tr": 0.0,
            "hist_std_Tr": 0.0,
            "hist_positive_rate": 0.0,
        }
    
    return {
        "hist_mean_Tr": float(prev_data.mean()),
        "hist_std_Tr": float(prev_data.std()) if len(prev_data) > 1 else 0.0,
        "hist_positive_rate": float((prev_data > 0).mean()),
    }


def compute_task_features(transition: str) -> dict[str, float]:
    """Compute task-level features from transition string."""
    task = int(transition.split("_")[0][1:])
    n_old = sum(len(TASK_CLASSES[t]) for t in range(1, task))
    n_new = len(TASK_CLASSES[task])
    
    return {
        "n_classes_old": float(n_old),
        "n_classes_new": float(n_new),
        "class_imbalance": float(n_new) / float(n_old) if n_old > 0 else 0.0,
        "transition_idx": float(task),
        "is_early_transition": 1.0 if task <= 3 else 0.0,
    }


def compute_relation_features(relation: str) -> dict[str, float]:
    """Compute relation-level features."""
    return {
        "rel_param_count": float(RELATION_PARAM_COUNTS.get(relation, 0)),
        "rel_layer_count": 3.0,  # 3-layer GNN
    }


def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: str = "boundary",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix X and target y from dataset.
    
    Args:
        df: Dataset with columns including T_r, S_r, D_r, G_r, transition, relation, seed
        feature_set: "boundary" (only boundary-available), "all" (all features), or list of feature names
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (T_r)
        feature_names: List of feature names in order
    """
    if feature_set == "boundary":
        feature_names = get_boundary_features()
    elif feature_set == "all":
        feature_names = get_all_feature_names()
    elif isinstance(feature_set, list):
        feature_names = feature_set
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    
    # Ensure all required columns exist
    required_cols = ["T_r", "S_r", "D_r", "G_r", "transition", "relation"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")
    
    X_rows = []
    y_rows = []
    
    for _, row in df.iterrows():
        transition = row["transition"]
        relation = row["relation"]
        
        features = {}
        
        # Core predictors
        if "S_r" in feature_names:
            features["S_r"] = float(row["S_r"]) if pd.notna(row["S_r"]) else 0.0
        if "D_r" in feature_names:
            features["D_r"] = float(row["D_r"]) if pd.notna(row["D_r"]) else 0.0
        if "G_r" in feature_names:
            features["G_r"] = float(row["G_r"]) if pd.notna(row["G_r"]) else 0.0
        
        # Task-level features
        task_feats = compute_task_features(transition)
        for k, v in task_feats.items():
            if k in feature_names:
                features[k] = v
        
        # Relation-level features
        rel_feats = compute_relation_features(relation)
        for k, v in rel_feats.items():
            if k in feature_names:
                features[k] = v
        
        # Historical features
        hist_feats = compute_historical_features(df, transition, relation)
        for k, v in hist_feats.items():
            if k in feature_names:
                features[k] = v
        
        # Interaction features
        if "S_r_x_D_r" in feature_names:
            s_r = features.get("S_r", 0.0)
            d_r = features.get("D_r", 0.0)
            features["S_r_x_D_r"] = s_r * d_r
        if "S_r_x_G_r" in feature_names:
            s_r = features.get("S_r", 0.0)
            g_r = features.get("G_r", 0.0)
            features["S_r_x_G_r"] = s_r * g_r
        if "D_r_x_G_r" in feature_names:
            d_r = features.get("D_r", 0.0)
            g_r = features.get("G_r", 0.0)
            features["D_r_x_G_r"] = d_r * g_r
        
        # Ensure all feature_names are present (fill missing with 0)
        X_row = [features.get(name, 0.0) for name in feature_names]
        X_rows.append(X_row)
        y_rows.append(float(row["T_r"]))
    
    return np.array(X_rows), np.array(y_rows), feature_names


def build_feature_matrix_with_provenance(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Build feature matrices for different provenance levels.
    
    Returns dict with:
    - boundary: (X, y, feature_names) - deployable features only
    - all: (X, y, feature_names) - all features including post-hoc
    """
    X_boundary, y_boundary, names_boundary = build_feature_matrix(df, "boundary")
    X_all, y_all, names_all = build_feature_matrix(df, "all")
    
    return {
        "boundary": {"X": X_boundary, "y": y_boundary, "feature_names": names_boundary},
        "all": {"X": X_all, "y": y_all, "feature_names": names_all},
    }


def print_feature_provenance_table():
    """Print a table of features with their provenance."""
    print(f"{'Feature':<25} {'Provenance':<15} Description")
    print("-" * 80)
    for spec in FEATURE_SPECS:
        print(f"{spec.name:<25} {spec.provenance.value:<15} {spec.description}")


if __name__ == "__main__":
    print_feature_provenance_table()