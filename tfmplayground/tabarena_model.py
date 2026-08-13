"""Optional AutoGluon model wrapper used by official TabArena runners."""

from __future__ import annotations

from typing import Any

import torch

from tfmplayground.bayesian_interface import BayesianNanoTabPFNClassifier, VanillaNanoTabPFNClassifier
from tfmplayground.task_posterior_interface import TaskPosteriorClassifier

APPLICABILITY_CONSTRAINTS = {
    "problem_types": ("binary", "multiclass"),
    "max_train_rows": 10_000,
    "max_features": 500,
    "max_classes": 10,
    "regression": False,
}

BAYESIAN_APPLICABILITY_CONSTRAINTS = {
    "problem_types": ("binary",),
    "max_train_rows": 10_000,
    "max_features": 500,
    "max_classes": 2,
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
            classifier_params["device"] = "cuda" if num_gpus and num_gpus > 0 and torch.cuda.is_available() else "cpu"
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


if AbstractModel is not None:

    class TabArenaBayesianNanoTabPFNModel(AbstractModel):
        """Official TabArena wrapper for the static Bayesian binary model."""

        ag_key = "BNANO"
        ag_name = "BayesianNanoTabPFN"
        _supported_problem_types = ["binary"]

        @classmethod
        def get_applicability_constraints(cls) -> dict[str, Any]:
            return dict(BAYESIAN_APPLICABILITY_CONSTRAINTS)

        def _fit(self, X, y, time_limit=None, num_gpus=0, **kwargs):
            if len(X) > BAYESIAN_APPLICABILITY_CONSTRAINTS["max_train_rows"]:
                raise ValueError("Static Bayesian nanoTabPFN supports at most 10,000 training rows.")
            if X.shape[1] > BAYESIAN_APPLICABILITY_CONSTRAINTS["max_features"]:
                raise ValueError("Static Bayesian nanoTabPFN supports at most 500 features.")
            if y.nunique() != 2:
                raise ValueError("Static Bayesian nanoTabPFN is binary-only in v1.")
            params = self._get_model_params()
            allowed = {"model", "num_hypotheses", "context_size", "random_state", "num_mem_chunks"}
            classifier_params = {key: value for key, value in params.items() if key in allowed}
            classifier_params["device"] = "cuda" if num_gpus and num_gpus > 0 and torch.cuda.is_available() else "cpu"
            self.model = BayesianNanoTabPFNClassifier(**classifier_params).fit(X, y)
            return self

        def _predict_proba(self, X, **kwargs):
            # TabArena's binary AutoGluon adapter widens a one-dimensional
            # positive-class vector to two columns itself.
            return self.model.predict_proba(X)[:, 1]

        def _set_default_params(self):
            for name, value in {"num_hypotheses": 2, "context_size": 1024, "random_state": 0}.items():
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

    class TabArenaBayesianNanoTabPFNModel:
        """Import-safe placeholder when AutoGluon is absent."""

        @classmethod
        def get_applicability_constraints(cls) -> dict[str, Any]:
            return dict(BAYESIAN_APPLICABILITY_CONSTRAINTS)

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TabArenaBayesianNanoTabPFNModel requires AutoGluon; install it in the official TabArena environment."
            )


if AbstractModel is not None:

    class TabArenaVanillaNanoTabPFNModel(AbstractModel):
        """Official TabArena wrapper for the unmodified nanoTabPFN baseline."""

        ag_key = "NANO"
        ag_name = "VanillaNanoTabPFN"
        _supported_problem_types = ["binary"]

        @classmethod
        def get_applicability_constraints(cls) -> dict[str, Any]:
            return dict(BAYESIAN_APPLICABILITY_CONSTRAINTS)

        def _fit(self, X, y, time_limit=None, num_gpus=0, **kwargs):
            if len(X) > 10_000 or X.shape[1] > 500 or y.nunique() != 2:
                raise ValueError(
                    "Vanilla nanoTabPFN comparison supports binary tasks with <=10,000 rows and <=500 features."
                )
            params = self._get_model_params()
            allowed = {"model", "context_size", "random_state", "num_mem_chunks"}
            classifier_params = {key: value for key, value in params.items() if key in allowed}
            classifier_params["device"] = "cuda" if num_gpus and num_gpus > 0 and torch.cuda.is_available() else "cpu"
            self.model = VanillaNanoTabPFNClassifier(**classifier_params).fit(X, y)
            return self

        def _predict_proba(self, X, **kwargs):
            return self.model.predict_proba(X)[:, 1]

        def _set_default_params(self):
            for name, value in {"context_size": 1024, "random_state": 0}.items():
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

    class TabArenaVanillaNanoTabPFNModel:
        """Import-safe placeholder when AutoGluon is absent."""

        @classmethod
        def get_applicability_constraints(cls) -> dict[str, Any]:
            return dict(BAYESIAN_APPLICABILITY_CONSTRAINTS)

        def __init__(self, *args, **kwargs):
            raise ImportError("TabArenaVanillaNanoTabPFNModel requires AutoGluon.")
