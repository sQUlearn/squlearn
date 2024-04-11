"""QNN module for classification and regression."""

from .loss import SquaredLoss, LogLoss, VarianceLoss, ConstantLoss, ParameterRegularizationLoss
from .qnnc import QNNClassifier
from .qnnr import QNNRegressor
from .training import get_variance_fac, get_lr_decay, ShotsFromRSTD

__all__ = [
    "SquaredLoss",
    "LogLoss",
    "VarianceLoss",
    "ConstantLoss",
    "ParameterRegularizationLoss",
    "QNNClassifier",
    "QNNRegressor",
    "get_variance_fac",
    "get_lr_decay",
    "ShotsFromRSTD",
]
