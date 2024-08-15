"""A library for quantum machine learning following the scikit-learnstandard."""

from .util import Executor
from . import observables, encoding_circuit, kernel, optimizers, qnn, util

__version__ = "0.7.6"

__all__ = [
    "Executor",
    "observables",
    "encoding_circuit",
    "kernel",
    "optimizers",
    "qnn",
    "util",
]
