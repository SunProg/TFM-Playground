"""Optional AutoGluon model wrapper used by official TabArena runners."""

from __future__ import annotations

from typing import Any

from tfmplayground.task_posterior_interface import TaskPosteriorClassifier

APPLICABILITY_CONSTRAINTS = {
    "problem_types": ("binary", "multiclass"),
    "max_train_rows": 10_000,
    "max_features": 500,
    "max_classes": 10,
    "regression": False,
}


try:
    from autogluon.core.models import AbstractModel
except ImportError:  # AutoGluon is deliberately not a core project dependency.
    AbstractModel = None


if AbstractModel is not None:

    class TabArenaTaskPosteriorModel(AbstractModel):
        """AutoGluon adapter with explicit classification-only constraints."""

        ag_key = "TPA"
        ag_name = "TaskPosteriorAdapter"
        _supported_problem_types = ["binary", "multiclass"]

        @classmethod
        def get_applicability_constraints(cls) -> dict[str, Any]:
            return dict(APPLICABILITY_CONSTRAINTS)

        def _fit(self, X, y, time_limit=None, num_gpus=0, **kwargs):
            if len(X) > APPLICABILITY_CONSTRAINTS["max_train_rows"]:
                raise ValueError("Task-posterior adapter supports at most 10,000 training rows.")
            if X.shape[1] > APPLICABILITY_CONSTRAINTS["max_features"]:
                raise ValueError("Task-posterior adapter supports at most 500 features.")
            if y.nunique() > APPLICABILITY_CONSTRAINTS["max_classes"]:
                raise ValueError("Task-posterior adapter supports at most 10 classes.")
            params = self._get_model_params()
            allowed = {
                "model",
                "particle_count",
                "context_size",
                "context_ensembles",
                "random_state",
                "context_mode",
                "query_chunk_size",
                "num_mem_chunks",
            }
            classifier_params = {key: value for key, value in params.items() if key in allowed}
            classifier_params["device"] = "cuda" if num_gpus and num_gpus > 0 else "cpu"
            self.model = TaskPosteriorClassifier(**classifier_params).fit(X, y)
            return self

        def _predict_proba(self, X, **kwargs):
            return self.model.predict_proba(X)

        def _set_default_params(self):
            defaults = {
                "particle_count": 4,
                "context_size": 1024,
                "context_ensembles": 4,
                "random_state": 0,
                "context_mode": "iid_set",
            }
            for name, value in defaults.items():
                self._set_default_param_value(name, value)

        def _get_default_resources(self) -> tuple[int, int]:
            return 1, 0

        def _more_tags(self):
            return {"can_refit_full": True, "valid_oof": False}

        @classmethod
        def supported_problem_types(cls) -> list[str]:
            return list(cls._supported_problem_types)

        @classmethod
        def config_generator(cls):
            from tabarena.utils.config_utils import ConfigGenerator

            return ConfigGenerator(model_cls=cls, manual_configs=[{}], search_space={})

else:

    class TabArenaTaskPosteriorModel:
        """Import-safe placeholder when the optional AutoGluon package is absent."""

        @classmethod
        def get_applicability_constraints(cls) -> dict[str, Any]:
            return dict(APPLICABILITY_CONSTRAINTS)

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TabArenaTaskPosteriorModel requires AutoGluon; install it in the official TabArena environment."
            )
