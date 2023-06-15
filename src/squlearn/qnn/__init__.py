"""QNN module for classification and regression."""
from .loss import SquaredLoss, VarianceLoss
from .qnnc import QNNClassifier
from .qnnr import QNNRegressor
from .training import get_variance_fac

__all__ = ["SquaredLoss", "VarianceLoss", "QNNClassifier", "QNNRegressor", "get_variance_fac"]
