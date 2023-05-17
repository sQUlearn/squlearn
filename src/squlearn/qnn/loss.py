"""Loss function implementations for QNNs."""
import abc
from typing import Union
import numpy as np


class LossBase(abc.ABC):
    """Base class implementation for Loss functions.

    This class is purely static. This means it doesn't have a __init__ funciton and doesnt need
    to be instanciated.
    """

    @property
    @abc.abstractmethod
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation.

        Returns:
            Evaluation tuple
        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def gradient_args_tuple(opt_param_op: bool = True) -> tuple:
        """Returns evaluation tuple for gradient calculation.

        Args:
            opt_param_op: True, if operator has trainable parameters

        Returns:
            Evaluation tuple
        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def value(value_dict: dict, **kwargs) -> float:
        """Calculates and returns the loss value."""
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def gradient(value_dict: dict, **kwargs) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Calculates and returns the gradient value."""
        raise NotImplementedError()


class SquaredLoss(LossBase):
    """Squared loss for regression."""

    loss_args_tuple = ("f",)

    @staticmethod
    def gradient_args_tuple(opt_param_op: bool = True) -> tuple:
        if opt_param_op:
            return ("f", "dfdp", "dfdop")
        return ("f", "dfdp")

    @staticmethod
    def value(value_dict: dict, **kwargs) -> float:
        r"""Calculates the squared loss.

        This function calculates the squared loss between the values in value_dict and ground_truth
        as
        .. math::
            \sum_i w_i \left|f\left(x_i\right)-f_ref\left(x_i\right)\right|^2

        Args:
            value_dict: Contains calculated values of the model
            ground_truth: The true values
            weights: Weight for each datapoint, if None all datapoints count the same

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

    @staticmethod
    def gradient(value_dict: dict, **kwargs) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        r"""Returns the gradient of the squared loss.

        This function calculates the gradient of the squared loss between the values in value_dict
        and ground_truth as
        .. math::
            2 * \sum_i w_i \left|f\left(x_i\right)-f_ref\left(x_i\right)\right|

        Args:
            value_dict: Contains calculated values of the model
            ground_truth: The true values
            weights: Weight for each datapoint, if None all datapoints count the same
            multiple_output: True if the qnn has multiple outputs
            opt_param_op: True if qnns operators have learnable parameters

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
        opt_param_op = "opt_param_op" in kwargs and kwargs["opt_param_op"]

        weighted_diff = np.multiply((value_dict["f"] - ground_truth), weights)
        if multiple_output:
            d_p = 2.0 * np.einsum("ij,ijk->k", weighted_diff, value_dict["dfdp"])
        else:
            d_p = 2.0 * np.einsum("j,jk->k", weighted_diff, value_dict["dfdp"])

        # Extra code for the cost operator derivatives
        if not opt_param_op:
            return d_p

        if multiple_output:
            d_op = 2.0 * np.einsum("ij,ijk->k", weighted_diff, value_dict["dfdop"])
        else:
            d_op = 2.0 * np.einsum("j,jk->k", weighted_diff, value_dict["dfdop"])
        return d_p, d_op


class VarianceLoss(LossBase):
    """Variance loss for regression."""

    loss_args_tuple = ("var",)

    @staticmethod
    def gradient_args_tuple(opt_param_op: bool = True) -> tuple:
        if opt_param_op:
            return ("var", "dvardp", "dvardop")
        return ("var", "dvardp")

    @staticmethod
    def value(value_dict: dict, **kwargs) -> float:
        r"""Returns the variance.

        This function returns the weighted variance as
        .. math::
            \lambda \sum_i \var_i

        Args:
            value_dict: Contains calculated values of the model
            variance_factor: multiplier lambda
            iteration: iteration number, if variance_factor is a function

        Returns:
            Loss value
        """
        if "variance_factor" not in kwargs:
            raise AttributeError("VarianceLoss requires variance_factor.")
        variance_factor = kwargs["variance_factor"]
        if callable(variance_factor):
            if "iteration" not in kwargs:
                raise AttributeError("If variance_facrot is callable, iteration is required.")
            variance_factor = variance_factor(kwargs["iteration"])
        return variance_factor * np.sum(value_dict["var"])

    @staticmethod
    def gradient(value_dict: dict, **kwargs) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Returns the gradient of the variance.

        This function calculates the gradient of the variance values in value_dict.

        Args:
            value_dict: Contains calculated values of the model
            variance_factor: multiplier lambda
            iteration: iteration number, if variance_factor is a function
            multiple_output: True if the qnn has multiple outputs
            opt_param_op: True if qnns operators have learnable parameters

        Returns:
            Gradient values
        """
        if "variance_factor" not in kwargs:
            raise AttributeError("VarianceLoss requires variance_factor.")
        variance_factor = kwargs["variance_factor"]
        if callable(variance_factor):
            if "iteration" not in kwargs:
                raise AttributeError("If variance_facrot is callable, iteration is required.")
            variance_factor = variance_factor(kwargs["iteration"])
        multiple_output = "multiple_output" in kwargs and kwargs["multiple_output"]
        opt_param_op = "opt_param_op" in kwargs and kwargs["opt_param_op"]

        if multiple_output:
            d_p = variance_factor * np.sum(value_dict["dvardp"], axis=(0, 1))
        else:
            d_p = variance_factor * np.sum(value_dict["dvardp"], axis=0)

        # Extra code for the cost operator derivatives
        if not opt_param_op:
            return d_p

        if multiple_output:
            d_op = variance_factor * np.sum(value_dict["dvardop"], axis=(0, 1))
        else:
            d_op = variance_factor * np.sum(value_dict["dvardop"], axis=0)

        return d_p, d_op
