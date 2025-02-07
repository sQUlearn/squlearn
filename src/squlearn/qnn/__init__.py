"""QNN module for classification and regression."""

from .loss import (
    ConstantLoss,
    LogLoss,
    ODELoss,
    ParameterRegularizationLoss,
    SquaredLoss,
    VarianceLoss,
)
from .qnnc import QNNClassifier
from .qnnr import QNNRegressor

__all__ = [
    "ConstantLoss",
    "LogLoss",
    "ODELoss",
    "ParameterRegularizationLoss",
    "SquaredLoss",
    "VarianceLoss",
    "QNNClassifier",
    "QNNRegressor",
]
