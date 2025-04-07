"""Parameter Regularization Loss for QNNs."""

from collections.abc import Callable
from typing import Union

import numpy as np

from .qnn_loss_base import QNNLossBase


class ParameterRegularizationLoss(QNNLossBase):
    r"""Loss for parameter regularization.

    Possible implementations:

    * ``"L1"``: :math:`L=\alpha \sum_i \left|p_i\right|`
    * ``"L2"``: :math:`L=\alpha \sum_i p_i^2`

    Args:
        alpha (float, Callable[[int], float]): Weight value :math:`\alpha`
        mode (str): Type of regularization, either 'L1' or 'L2' (default: 'L2').
        parameter_list (list): List of parameters to regularize, None: all (default: None).
        parameter_operator_list (list): List of operator parameters to regularize, None: all
            (default: []).
    """

    def __init__(
        self,
        alpha: Union[float, Callable[[int], float]] = 0.005,
        mode: str = "L2",
        parameter_list: Union[list, None] = None,
        parameter_operator_list: Union[list, None] = None,
    ):
        super().__init__()
        self._alpha = alpha
        self._mode = mode
        if self._mode not in ["L1", "L2"]:
            raise ValueError("Type must be 'L1' or 'L2'!")

        self._parameter_list = parameter_list
        self._parameter_operator_list = parameter_operator_list

    @property
    def loss_variance_available(self) -> bool:
        """Returns True since variance is zero (and available)."""
        return True

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation."""
        return tuple()

    @property
    def variance_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation."""
        return tuple()

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss gradient calculation."""
        return tuple()

    def value(self, value_dict: dict, **kwargs) -> float:
        r"""Returns the variance.

        This function returns the weighted variance as

        .. math::
            L_\text{var} = \alpha \sum_i \var_i

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

        loss = 0.0
        if self._parameter_list is None:
            if self._mode == "L1":
                loss += np.sum(np.abs(value_dict["param"]))
            elif self._mode == "L2":
                loss += np.sum(np.square(value_dict["param"]))
            else:
                raise ValueError("Type must be L1 or L2!")
        else:
            if self._mode == "L1":
                loss += np.sum(np.abs(value_dict["param"][self._parameter_list]))
            elif self._mode == "L2":
                loss += np.sum(np.square(value_dict["param"][self._parameter_list]))
            else:
                raise ValueError("Type must be L1 or L2!")

        if self._opt_param_op:
            if self._parameter_list is None:
                if self._mode == "L1":
                    loss += np.sum(np.abs(value_dict["param_op"]))
                elif self._mode == "L2":
                    loss += np.sum(np.square(value_dict["param_op"]))
                else:
                    raise ValueError("Type must be L1 or L2!")
            else:
                if self._mode == "L1":
                    loss += np.sum(np.abs(value_dict["param_op"][self._parameter_operator_list]))
                elif self._mode == "L2":
                    loss += np.sum(
                        np.square(value_dict["param_op"][self._parameter_operator_list])
                    )
                else:
                    raise ValueError("Type must be L1 or L2!")

        return alpha * loss

    def variance(self, value_dict: dict, **kwargs) -> float:
        """Returns 0 since the variance is equal to zero."""
        return 0.0

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Returns the gradient of the variance.

        This function calculates the gradient of the variance values in value_dict.

        Args:
            value_dict (dict): Contains calculated values of the model
            iteration (int): iteration number, if variance_factor is a function

        Returns:
            Gradient values
        """
        if callable(self._alpha):
            if "iteration" not in kwargs:
                raise AttributeError("If alpha is callable, iteration is required.")
            alpha = self._alpha(kwargs["iteration"])
        else:
            alpha = self._alpha

        d_p = np.zeros_like(value_dict["param"])
        if self._parameter_list is None:
            if self._mode == "L1":
                d_p = alpha * np.sign(value_dict["param"])
            elif self._mode == "L2":
                d_p = alpha * 2.0 * value_dict["param"]
            else:
                raise ValueError("Type must be L1 or L2!")
        else:
            if self._mode == "L1":
                d_p[self._parameter_list] = alpha * np.sign(
                    value_dict["param"][self._parameter_list]
                )
            elif self._mode == "L2":
                d_p[self._parameter_list] = alpha * 2.0 * value_dict["param"][self._parameter_list]
            else:
                raise ValueError("Type must be L1 or L2!")

        # Extra code for the cost operator derivatives
        if not self._opt_param_op:
            return d_p

        d_op = np.zeros_like(value_dict["param_op"])
        if self._parameter_operator_list is None:
            if self._mode == "L1":
                d_op = alpha * np.sign(value_dict["param_op"])
            elif self._mode == "L2":
                d_op = alpha * 2.0 * value_dict["param_op"]
            else:
                raise ValueError("Type must be L1 or L2!")
        else:
            if self._mode == "L1":
                d_op[self._parameter_operator_list] = alpha * np.sign(
                    value_dict["param_op"][self._parameter_operator_list]
                )
            elif self._mode == "L2":
                d_op[self._parameter_operator_list] = (
                    alpha * 2.0 * value_dict["param_op"][self._parameter_operator_list]
                )
            else:
                raise ValueError("Type must be L1 or L2!")

        return d_p, d_op
