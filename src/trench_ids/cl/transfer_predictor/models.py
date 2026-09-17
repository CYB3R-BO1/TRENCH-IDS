"""Model definitions for transfer predictor."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ConstantRegressor(BaseEstimator, RegressorMixin):
    """Constant predictor that always predicts the mean of training targets."""
    
    def __init__(self):
        self.mean_ = 0.0
    
    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self
    
    def predict(self, X):
        return np.full(X.shape[0], self.mean_)
    
    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))


def get_model(model_name: str, random_state: int = 42) -> BaseEstimator:
    """Get model by name."""
    
    models = {
        "constant": ConstantRegressor(),
        
        "ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0, random_state=random_state)
        ),
        
        "ridge_tuned": make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0, random_state=random_state)
        ),
        
        "rf": RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        
        "rf_small": RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        
        "gb": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=random_state,
        ),
        
        "gb_small": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.1,
            min_samples_split=3,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=random_state,
        ),
        
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                max_iter=1000,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
            )
        ),
        
        "mlp_small": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=0.01,
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
            )
        ),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]


def get_all_model_names() -> list[str]:
    """Get all available model names."""
    return list(get_model.__annotations__.get("return", {}).keys()) if False else [
        "constant", "ridge", "ridge_tuned", "rf", "rf_small", "gb", "gb_small", "mlp", "mlp_small"
    ]


def get_model_config(model_name: str) -> dict[str, Any]:
    """Get configuration dict for a model (for logging/reproducibility)."""
    configs = {
        "constant": {"type": "ConstantRegressor"},
        "ridge": {"type": "Ridge", "alpha": 1.0, "scaler": "StandardScaler"},
        "ridge_tuned": {"type": "Ridge", "alpha": 10.0, "scaler": "StandardScaler"},
        "rf": {"type": "RandomForestRegressor", "n_estimators": 200, "max_depth": 5, "max_features": "sqrt"},
        "rf_small": {"type": "RandomForestRegressor", "n_estimators": 100, "max_depth": 3, "max_features": "sqrt"},
        "gb": {"type": "GradientBoostingRegressor", "n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        "gb_small": {"type": "GradientBoostingRegressor", "n_estimators": 100, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.8},
        "mlp": {"type": "MLPRegressor", "hidden_layers": [64, 32], "alpha": 0.001, "early_stopping": True},
        "mlp_small": {"type": "MLPRegressor", "hidden_layers": [32, 16], "alpha": 0.01, "early_stopping": True},
    }
    return configs.get(model_name, {"type": "Unknown"})


if __name__ == "__main__":
    for name in get_all_model_names():
        model = get_model(name)
        config = get_model_config(name)
        print(f"{name}: {config}")