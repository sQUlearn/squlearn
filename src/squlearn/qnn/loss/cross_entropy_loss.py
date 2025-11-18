"""Log Loss for QNNs."""

from typing import Union

import numpy as np

from .qnn_loss_base import QNNLossBase


class CrossEntropyLoss(QNNLossBase):
    r"""Cross entropy loss for classification.

    Args:
        eps (float): Small value to avoid :math:`\log(0)`

    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self._eps = eps

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for the log loss calculation."""
        return ("f",)

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for the log loss gradient calculation."""
        if self._opt_param_op:
            return ("f", "dfdp", "dfdop")
        return ("f", "dfdp")

    def value(self, value_dict: dict, **kwargs) -> float:
        r"""Calculates the cross entropy loss.

        This function calculates the cross entropy loss between the probability values in
        `value_dict` and binary `ground_truth` as

        .. math::
            - \left(\sum_i w_i \cdot \sum_k y_{i,k} \cdot \log\left(f\left(x_i\right)_k\right)
                \right)

        Args:
            value_dict (dict): Contains calculated values of the model
            ground_truth (np.ndarray): The true values :math:`y\left(x_i\right)`
            weights (np.ndarray): Weight for each data point, if None all data points count the
                same

        Returns:
            Loss value
        """
        if "ground_truth" not in kwargs:
            raise AttributeError("CrossEntropyLoss requires ground_truth.")

        ground_truth = kwargs["ground_truth"]
        weights = kwargs.get("weights") or np.ones_like(ground_truth)

        probability_values = np.clip(value_dict["f"], self._eps, 1.0 - self._eps)
        if probability_values.ndim == 1:
            probability_values = np.stack([probability_values, 1.0 - probability_values], axis=1)
            ground_truth = np.stack([ground_truth, 1.0 - ground_truth], axis=1)
            weights = np.tile(weights.reshape(-1, 1), 2)

        loss = -1.0 * np.mean(
            np.sum(np.multiply(ground_truth * np.log(probability_values), weights), axis=1)
        )
        return loss

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        r"""Returns the gradient of the cross entropy loss.

        This function calculates the gradient of the cross entropy loss between the probability
        values in `value_dict` and binary `ground_truth` as

        .. math::
            - \left(\sum_i w_i \cdot \left(\frac{y_i}{f\left(x_i\right)}
                - \frac{(1-y_i)}{1-f\left(x_i\right)}\right)
                \cdot \frac{\partial f\left(x_i\right)}{\partial \theta} \right)

        Args:
            value_dict (dict): Contains calculated values of the model
            ground_truth (np.ndarray): The true values :math:`y\left(x_i\right)`
            weights (np.ndarray): Weight for each data point, if None all data points count the
                same
            multiple_output (bool): True if the QNN has multiple outputs

        Returns:
            Gradient values
        """
        if "ground_truth" not in kwargs:
            raise AttributeError("CrossEntropyLoss requires ground_truth.")

        ground_truth = kwargs["ground_truth"]
        weights = kwargs.get("weights") or np.ones_like(ground_truth)
        multiple_output = kwargs.get("multiple_output", False)

        probability_values = np.clip(value_dict["f"], self._eps, 1.0 - self._eps)
        binary = probability_values.ndim == 1
        if binary:
            probability_values = np.stack([probability_values, probability_values - 1.0], axis=1)
            ground_truth = np.stack([ground_truth, 1.0 - ground_truth], axis=1)
            weights = np.tile(weights.reshape(-1, 1), 2)

        weighted_outer_gradient = np.multiply(
            # ground_truth / probability_values,
            (ground_truth / probability_values - (1 - ground_truth) / (1 - probability_values)),
            weights,
        )

        if binary:
            weighted_outer_gradient = np.sum(weighted_outer_gradient, axis=1)

        if multiple_output:
            d_p = (
                -1.0
                * np.einsum("ij,ijk->k", weighted_outer_gradient, value_dict["dfdp"])
                / ground_truth.shape[0]
            )
        else:
            d_p = (
                -1.0
                * np.einsum("j,jk->k", weighted_outer_gradient, value_dict["dfdp"])
                / ground_truth.shape[0]
            )

        # Extra code for the cost operator derivatives
        if not self._opt_param_op:
            return d_p

        if multiple_output:
            d_op = (
                -1.0
                * np.einsum("ij,ijk->k", weighted_outer_gradient, value_dict["dfdop"])
                / ground_truth.shape[0]
            )
        else:
            d_op = (
                -1.0
                * np.einsum("j,jk->k", weighted_outer_gradient, value_dict["dfdop"])
                / ground_truth.shape[0]
            )
        return d_p, d_op
