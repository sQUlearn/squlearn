"""A library for quantum machine learning following the sklearn standard."""
from .util import Executor
from . import expectation_operator, feature_map, kernel, optimizers, qnn, util

__version__ = "0.2.0"

__all__ = [
    "Executor",
    "expectation_operator",
    "feature_map",
    "kernel",
    "optimizers",
    "qnn",
    "util",
]
