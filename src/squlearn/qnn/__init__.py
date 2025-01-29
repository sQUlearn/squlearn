"""QNN module for classification and regression."""

from .loss import ConstantLoss, ODELoss, ParameterRegularizationLoss, SquaredLoss, VarianceLoss
from .qnnc import QNNClassifier
from .qnnr import QNNRegressor
from .training import get_variance_fac, get_lr_decay, ShotsFromRSTD

__all__ = [
    "ConstantLoss",
    "ODELoss",
    "ParameterRegularizationLoss",
    "SquaredLoss",
    "VarianceLoss",
    "QNNClassifier",
    "QNNRegressor",
    "get_variance_fac",
    "get_lr_decay",
    "ShotsFromRSTD",
]
