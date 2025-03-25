"""Squared Loss for QNNs."""

from typing import Union

import numpy as np

from .qnn_loss_base import QNNLossBase


class SquaredLoss(QNNLossBase):
    """Squared loss for regression."""

    @property
    def loss_variance_available(self) -> bool:
        """Returns True since the squared loss function has a variance function."""
        return True

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for the squared loss calculation."""
        return ("f",)

    @property
    def variance_args_tuple(self) -> tuple:
        """Returns evaluation tuple for the squared loss variance calculation."""
        return ("f", "var")

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for the squared loss gradient calculation."""
        if self._opt_param_op:
            return ("f", "dfdp", "dfdop")
        return ("f", "dfdp")

    def value(self, value_dict: dict, **kwargs) -> float:
        r"""Calculates the squared loss.

        This function calculates the squared loss between the values in value_dict and ground_truth
        as

        .. math::
            \sum_i w_i \left|f\left(x_i\right)-f_ref\left(x_i\right)\right|^2

        Args:
            value_dict (dict): Contains calculated values of the model
            ground_truth (np.ndarray): The true values :math:`f_ref\left(x_i\right)`
            weights (np.ndarray): Weight for each data point, if None all data points count the
            same

        Returns:
            Loss value
        """
        if "ground_truth" not in kwargs:
            raise AttributeError("SquaredLoss requires ground_truth.")
        ground_truth = kwargs["ground_truth"]
        if "weights" in kwargs and kwargs["weights"] is not None:
            weights = kwargs["weights"]
        else:
            weights = np.ones_like(ground_truth)
        return np.sum(np.multiply(np.square(value_dict["f"] - ground_truth), weights))

    def variance(self, value_dict: dict, **kwargs) -> float:
        r"""Calculates the approximated variance of the squared loss.

        This function calculates the approximated variance of the squared loss

        .. math::
            4\sum_i w_i \left|f\left(x_i\right)-f_ref\left(x_i\right)\right|^2 \sigma_f^2(x_i)

        Args:
            value_dict (dict): Contains calculated values of the model
            ground_truth (np.ndarray): The true values :math:`f_ref\left(x_i\right)`
            weights (np.ndarray): Weight for each data point, if None all data points count the
                same

        Returns:
            Loss value
        """
        if "ground_truth" not in kwargs:
            raise AttributeError("SquaredLoss requires ground_truth.")
        ground_truth = kwargs["ground_truth"]
        if "weights" in kwargs and kwargs["weights"] is not None:
            weights = kwargs["weights"]
        else:
            weights = np.ones_like(ground_truth)

        diff_square = np.multiply(weights, np.square(value_dict["f"] - ground_truth))
        return np.sum(4 * np.multiply(diff_square, value_dict["var"]))

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        r"""Returns the gradient of the squared loss.

        This function calculates the gradient of the squared loss between the values in value_dict
        and ground_truth as

        .. math::
           2\sum_j \sum_i w_i \left(f\left(x_i\right)-f_ref\left(x_i\right)\right)
           \frac{\partial f(x_i)}{\partial p_j}

        Args:
            value_dict (dict): Contains calculated values of the model
            ground_truth (np.ndarray): The true values :math:`f_ref\left(x_i\right)`
            weights (np.ndarray): Weight for each data point, if None all data points count the
                same
            multiple_output (bool): True if the QNN has multiple outputs

        Returns:
            Gradient values
        """

        if "ground_truth" not in kwargs:
            raise AttributeError("SquaredLoss requires ground_truth.")

        ground_truth = kwargs["ground_truth"]
        if "weights" in kwargs and kwargs["weights"] is not None:
            weights = kwargs["weights"]
        else:
            weights = np.ones_like(ground_truth)
        multiple_output = "multiple_output" in kwargs and kwargs["multiple_output"]

        weighted_diff = np.multiply((value_dict["f"] - ground_truth), weights)

        if value_dict["dfdp"].shape[0] == 0:
            d_p = np.array([])
        else:
            if multiple_output:
                d_p = 2.0 * np.einsum("ij,ijk->k", weighted_diff, value_dict["dfdp"])
            else:
                d_p = 2.0 * np.einsum("j,jk->k", weighted_diff, value_dict["dfdp"])

        # Extra code for the cost operator derivatives
        if not self._opt_param_op:
            return d_p

        if value_dict["dfdop"].shape[0] == 0:
            d_op = np.array([])
        else:
            if multiple_output:
                d_op = 2.0 * np.einsum("ij,ijk->k", weighted_diff, value_dict["dfdop"])
            else:
                d_op = 2.0 * np.einsum("j,jk->k", weighted_diff, value_dict["dfdop"])
        return d_p, d_op
