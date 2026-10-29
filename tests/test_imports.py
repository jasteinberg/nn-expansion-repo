"""Every module of the package and the experiments imports (no broken leftovers)."""
from __future__ import annotations

import importlib
import pkgutil

import pytest

import experiments
import nn_expansion

MODULES = [m.name for pkg in (nn_expansion, experiments)
           for m in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + ".")]


def test_modules_found() -> None:
    assert "nn_expansion.svm" in MODULES and "experiments.reproduce" in MODULES


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
