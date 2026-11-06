"""
Model Factory for Tabular-Gauge.
Responsible for lazy loading and instantiating model wrappers based on configuration.
"""

import importlib
import logging
from typing import Dict, Any, List, Optional, Type
from .base import ModelWrapper

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Centralized factory to instantiate model wrappers.
    Supports lazy imports to prevent crashes when optional dependencies are missing.
    """

    # Mapping of model identifiers in YAML to their Python class paths.
    # format: "model_id": "module.path.ClassName"
    _REGISTRY = {
        "tabpfn": "tabular_gauge.models.tabpfn_model.TabPFNWrapper",
        "xgboost": "tabular_gauge.models.xgboost_model.XGBoostWrapper",
        "lightgbm": "tabular_gauge.models.lightgbm_model.LightGBMWrapper",
        "catboost": "tabular_gauge.models.catboost_model.CatBoostWrapper",
        "mlp_sklearn": "tabular_gauge.models.mlp_model.MLPWrapper",
    }

    @classmethod
    def register_model(cls, model_id: str, class_path: str):
        """
        Dynamically register a new model wrapper.

        Args:
            model_id: Unique identifier for the model (e.g., 'my_custom_model')
            class_path: Full Python path to the wrapper class (e.g., 'my_module.MyModelWrapper')
        """
        cls._REGISTRY[model_id] = class_path
        logger.info(f"Registered model '{model_id}' at {class_path}")

    @classmethod
    def create(cls, model_id: str, config: Dict[str, Any], seed: int) -> ModelWrapper:
        """
        Instantiate a model wrapper based on the model_id and provided configuration.

        Args:
            model_id: Identifier from models.yaml
            config: Configuration dict for the specific model
            seed: Random state to propagate

        Returns:
            An initialized ModelWrapper instance.

        Raises:
            ValueError: If the model_id is not registered.
            ImportError: If the required library for the model is not installed.
        """
        if model_id not in cls._REGISTRY:
            raise ValueError(f"Model '{model_id}' is not registered in the ModelFactory. "
                             f"Available models: {list(cls._REGISTRY.keys())}")

        class_path = cls._REGISTRY[model_id]
        module_path, class_name = class_path.rsplit(".", 1)

        try:
            # Lazy import of the module
            module = importlib.import_module(module_path)
            wrapper_class = getattr(module, class_name)
        except ImportError as e:
            # Provide a helpful error message for missing dependencies
            # We try to extract the missing package name from the ImportError
            missing_pkg = "unknown"
            if hasattr(e, 'name'):
                missing_pkg = e.name

            raise ImportError(
                f"Could not instantiate model '{model_id}' because a required dependency is missing: {missing_pkg}. "
                f"Please install the necessary package (e.g., 'pip install {missing_pkg}') or "
                f"disable this model in configs/models.yaml."
            ) from e

        # Extract parameters from config.
        # Convention: Use 'fixed_params' for static settings,
        # and check for top-level config keys that match constructor arguments.
        params = {}

        # 1. Add random_state first
        params["random_state"] = seed

        # 2. Add fixed_params if they exist
        if "fixed_params" in config:
            params.update(config["fixed_params"])

        # 3. Add other top-level config keys (e.g., n_trials)
        # We only add keys that are not 'class', 'tunable', 'description', 'fixed_params', 'search_space', 'enabled', 'id'
        for k, v in config.items():
            if k not in ("class", "tunable", "description", "fixed_params", "search_space", "enabled", "id"):
                params[k] = v

        try:
            return wrapper_class(**params)
        except TypeError as e:
            raise TypeError(
                f"Configuration mismatch for model '{model_id}'. The wrapper constructor "
                f"received unexpected or missing arguments: {e}. "
                f"Check configs/models.yaml against {class_path}.__init__"
            ) from e
