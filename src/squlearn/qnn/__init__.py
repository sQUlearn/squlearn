"""QNN module for classification and regression."""

from .loss import ConstantLoss, ODELoss, ParameterRegularizationLoss, SquaredLoss, VarianceLoss
from .qnnc import QNNClassifier
from .qnnr import QNNRegressor

__all__ = [
    "ConstantLoss",
    "ODELoss",
    "ParameterRegularizationLoss",
    "SquaredLoss",
    "VarianceLoss",
    "QNNClassifier",
    "QNNRegressor",
]
