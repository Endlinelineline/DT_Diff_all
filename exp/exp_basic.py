"""Base experiment with dynamic top-level model discovery."""

from __future__ import annotations

import importlib
import pkgutil

import models


class Exp_Basic:
    """Resolve ``--model`` to ``models/<ModelName>.py::Model``."""

    def __init__(self, args) -> None:
        self.args = args
        self.model_dict = self._discover_models()
        try:
            self.model_module = self.model_dict[args.model]
        except KeyError as exc:
            available = ", ".join(sorted(self.model_dict)) or "<none>"
            raise ValueError(
                f"Unknown model {args.model!r}. Available models: {available}"
            ) from exc

    @staticmethod
    def _discover_models() -> dict[str, object]:
        discovered = {}
        for module_info in pkgutil.iter_modules(models.__path__):
            if module_info.ispkg or module_info.name.startswith("_"):
                continue
            module = importlib.import_module(f"models.{module_info.name}")
            model_class = getattr(module, "Model", None)
            if model_class is not None:
                discovered[module_info.name] = module
        return discovered

    @property
    def model_class(self):
        return self.model_module.Model
