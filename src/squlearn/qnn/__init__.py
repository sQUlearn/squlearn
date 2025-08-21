"""QNN module for classification and regression."""

from .loss import (
    ConstantLoss,
    CrossEntropyLoss,
    ODELoss,
    ParameterRegularizationLoss,
    SquaredLoss,
    VarianceLoss,
)
from .qnnc import QNNClassifier
from .qnnr import QNNRegressor

__all__ = [
    "ConstantLoss",
    "CrossEntropyLoss",
    "ODELoss",
    "ParameterRegularizationLoss",
    "SquaredLoss",
    "VarianceLoss",
    "QNNClassifier",
    "QNNRegressor",
]
