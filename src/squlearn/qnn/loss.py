"""Loss function implementations for QNNs."""
import abc
from typing import Union
import numpy as np


class LossBase(abc.ABC):
    """Base class implementation for Loss functions.

    This class is purely static. This means it doesn't have a __init__ funciton and doesnt need
    to be instanciated.
    """

    def __init__(self):
        self._opt_param_op = True

    def set_optimize_param_op(self, opt_param_op: bool = True):
        """Sets the optimize_param_op flag.

        Args:
            opt_param_op: True, if operator has trainable parameters
        """
        self._opt_param_op = opt_param_op

    @property
    @abc.abstractmethod
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation.

        Returns:
            Evaluation tuple
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for gradient calculation.

        Args:
            opt_param_op: True, if operator has trainable parameters

        Returns:
            Evaluation tuple
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def value(self, value_dict: dict, **kwargs) -> float:
        """Calculates and returns the loss value."""
        raise NotImplementedError()

    @abc.abstractmethod
    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Calculates and returns the gradient of the loss."""
        raise NotImplementedError()

    def __add__(self, x):
        """Adds two loss functions together."""
        if not isinstance(x, LossBase):
            raise ValueError("Only the addition with other feature maps is allowed!")

        class AddedLoss(LossBase):
            """Special class for composed loss functions."""

            def __init__(self, l1, l2):
                super().__init__()
                self._l1 = l1
                self._l2 = l2

            def set_optimize_param_op(self, opt_param_op: bool = True):
                """Sets the optimize_param_op flag.

                Args:
                    opt_param_op: True, if operator has trainable parameters
                """
                self._l1.set_optimize_param_op(opt_param_op)
                self._l2.set_optimize_param_op(opt_param_op)

            @property
            def loss_args_tuple(self) -> tuple:
                """Returns evaluation tuple for loss calculation.

                Returns:
                    Evaluation tuple
                """
                return tuple(set(self._l1.loss_args_tuple + self._l2.loss_args_tuple))

            @property
            def gradient_args_tuple(self) -> tuple:
                """Returns evaluation tuple for gradient calculation.

                Args:
                    opt_param_op: True, if operator has trainable parameters

                Returns:
                    Evaluation tuple
                """
                return tuple(set(self._l1.gradient_args_tuple + self._l2.gradient_args_tuple))

            def value(self, value_dict: dict, **kwargs) -> float:
                return self._l1.value(value_dict, **kwargs) + self._l2.value(value_dict, **kwargs)

            @staticmethod
            def gradient(
                self, value_dict: dict, **kwargs
            ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
                return self._l1.gradient(value_dict, **kwargs) + self._l2.gradient(
                    value_dict, **kwargs
                )

        return AddedLoss(self, x)


class SquaredLoss(LossBase):
    """Squared loss for regression."""

    def __init__(self):
        super().__init__()

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation.

        Returns:
            Evaluation tuple
        """
        return ("f",)

    @property
    def gradient_args_tuple(self) -> tuple:
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

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
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

        weighted_diff = np.multiply((value_dict["f"] - ground_truth), weights)
        if multiple_output:
            d_p = 2.0 * np.einsum("ij,ijk->k", weighted_diff, value_dict["dfdp"])
        else:
            d_p = 2.0 * np.einsum("j,jk->k", weighted_diff, value_dict["dfdp"])

        # Extra code for the cost operator derivatives
        if not self._opt_param_op:
            return d_p

        if multiple_output:
            d_op = 2.0 * np.einsum("ij,ijk->k", weighted_diff, value_dict["dfdop"])
        else:
            d_op = 2.0 * np.einsum("j,jk->k", weighted_diff, value_dict["dfdop"])
        return d_p, d_op


class VarianceLoss(LossBase):
    """Variance loss for regression."""

    def __init__(self, alpha=0.005):
        super().__init__()
        self._alpha = alpha

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation.

        Returns:
            Evaluation tuple
        """
        return ("var",)

    @property
    def gradient_args_tuple(self) -> tuple:
        if self._opt_param_op:
            return ("var", "dvardp", "dvardop")
        return ("var", "dvardp")

    def value(self, value_dict: dict, **kwargs) -> float:
        r"""Returns the variance.

        This function returns the weighted variance as
        .. math::
            L_\text{var} = \alpha \sum_i \var_i

        Args:
            value_dict: Contains calculated values of the model
            iteration: iteration number, if alpha is a callable function

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

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Returns the gradient of the variance.

        This function calculates the gradient of the variance values in value_dict.

        Args:
            value_dict: Contains calculated values of the model
            iteration: iteration number, if variance_factor is a function
            multiple_output: True if the qnn has multiple outputs

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

        if multiple_output:
            d_p = alpha * np.sum(value_dict["dvardp"], axis=(0, 1))
        else:
            d_p = alpha * np.sum(value_dict["dvardp"], axis=0)

        # Extra code for the cost operator derivatives
        if not self._opt_param_op:
            return d_p

        if multiple_output:
            d_op = alpha * np.sum(value_dict["dvardop"], axis=(0, 1))
        else:
            d_op = alpha * np.sum(value_dict["dvardop"], axis=0)

        return d_p, d_op
