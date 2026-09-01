from typing import Union

import numpy as np

from qc_executor import Parameters
from qc_executor.utils.data_preprocessing import (
    _adjust_input,
    adjust_features,
    adjust_parameters,
    convert_to_float64,
    to_tuple,
)

__all__ = [
    "_adjust_input",
    "adjust_features",
    "adjust_parameters",
    "convert_to_float64",
    "extract_num_features",
    "to_tuple",
]


def extract_num_features(X: Union[np.ndarray, Parameters, list]) -> int:
    """Extract the number of features from the input array."""
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            return X.shape[0]
        elif X.ndim >= 2:
            return X.shape[1]
    if isinstance(X, Parameters):
        return len(X)

    raise TypeError("Unsupported type for X")
