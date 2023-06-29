"""QNN module for classification and regression."""
from .loss import SquaredLoss, VarianceLoss, ConstantLoss
from .qnnc import QNNClassifier
from .qnnr import QNNRegressor
from .training import get_variance_fac

__all__ = [
    "SquaredLoss",
    "VarianceLoss",
    "ConstantLoss",
    "QNNClassifier",
    "QNNRegressor",
    "get_variance_fac",
]
