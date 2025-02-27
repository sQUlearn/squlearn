"""Variance Loss for QNNs."""

from collections.abc import Callable
from typing import Union

import numpy as np

from .qnn_loss_base import QNNLossBase


class VarianceLoss(QNNLossBase):
    r"""Variance loss for regression.

    Args:
        alpha (float, Callable[[int], float]): Weight value :math:`\alpha`
    """

    def __init__(self, alpha: Union[float, Callable[[int], float]] = 0.005):
        super().__init__()
        self._alpha = alpha

    @property
    def loss_variance_available(self) -> bool:
        """Returns True since we neglect the variance of the variance."""
        return True

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation."""
        return ("var",)

    @property
    def variance_args_tuple(self) -> tuple:
        """Returns evaluation tuple for variance calculation."""
        return tuple()

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss gradient calculation."""
        if self._opt_param_op:
            return ("var", "dvardp", "dvardop")
        return ("var", "dvardp")

    def value(self, value_dict: dict, **kwargs) -> float:
        r"""Returns the variance.

        This function returns the weighted variance as

        .. math::
            L_\operatorname{Var} = \alpha \sum_i \operatorname{Var}_i

        Args:
            value_dict (dict): Contains calculated values of the model
            iteration (int): iteration number, if alpha is a callable function

        Returns:
            Loss value
        """

        if callable(self._alpha):
            if "iteration" not in kwargs:
                raise AttributeError("If alpha is callable, iteration is required.")
            alpha = self._alpha(kwargs["iteration"])
        else:
            alpha = self._alpha

        return alpha * np.sum(value_dict["var"])

    def variance(self, value_dict: dict, **kwargs) -> float:
        """Returns 0 since we neglect the variance of the variance."""
        return 0.0

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Returns the gradient of the variance.

        This function calculates the gradient of the variance values in value_dict.

        Args:
            value_dict (dict): Contains calculated values of the model
            iteration (int): iteration number, if variance_factor is a function
            multiple_output (bool): True if the QNN has multiple outputs

        Returns:
            Gradient values
        """
        if callable(self._alpha):
            if "iteration" not in kwargs:
                raise AttributeError("If alpha is callable, iteration is required.")
            alpha = self._alpha(kwargs["iteration"])
        else:
            alpha = self._alpha

        multiple_output = "multiple_output" in kwargs and kwargs["multiple_output"]
        if value_dict["dfdp"].shape[0] == 0:
            d_p = np.array([])
        else:
            if multiple_output:
                d_p = alpha * np.sum(value_dict["dvardp"], axis=(0, 1))
            else:
                d_p = alpha * np.sum(value_dict["dvardp"], axis=0)

        # Extra code for the cost operator derivatives
        if not self._opt_param_op:
            return d_p

        if value_dict["dfdop"].shape[0] == 0:
            d_op = np.array([])
        else:
            if multiple_output:
                d_op = alpha * np.sum(value_dict["dvardop"], axis=(0, 1))
            else:
                d_op = alpha * np.sum(value_dict["dvardop"], axis=0)

        return d_p, d_op
