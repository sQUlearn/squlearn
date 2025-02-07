"""Loss functions for QNNs."""

from .log_loss import LogLoss
from .ode_loss import ODELoss
from .parameter_regularization_loss import ParameterRegularizationLoss
from .qnn_loss_base import ConstantLoss
from .squared_loss import SquaredLoss
from .variance_loss import VarianceLoss

__all__ = [
    "ConstantLoss",
    "LogLoss",
    "ODELoss",
    "ParameterRegularizationLoss",
    "SquaredLoss",
    "VarianceLoss",
]
