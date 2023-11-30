"""Loss function implementations for QNNs."""
import abc
from typing import Union
import numpy as np
from collections.abc import Callable


class LossBase(abc.ABC):
    """Base class implementation for loss functions."""

    def __init__(self):
        self._opt_param_op = True

    def set_opt_param_op(self, opt_param_op: bool = True):
        """Sets the `opt_param_op` flag.

        Args:
            opt_param_op (bool): True, if operator has trainable parameters
        """
        self._opt_param_op = opt_param_op

    @property
    def loss_variance_available(self) -> bool:
        """Returns True if the loss function has a variance function."""
        return False

    @property
    @abc.abstractmethod
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation."""
        raise NotImplementedError()

    @property
    def variance_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss gradient calculation."""
        raise NotImplementedError()

    @abc.abstractmethod
    def value(self, value_dict: dict, **kwargs) -> float:
        """Calculates and returns the loss value."""
        raise NotImplementedError()

    def variance(self, value_dict: dict, **kwargs) -> float:
        """Calculates and returns the variance of the loss value."""
        raise NotImplementedError()

    @abc.abstractmethod
    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Calculates and returns the gradient of the loss."""
        raise NotImplementedError()

    def __add__(self, x):
        """Adds two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(self, x, "+")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(self, ConstantLoss(x), "+")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __radd__(self, x):
        """Adds two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(x, self, "+")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(ConstantLoss(x), self, "+")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __mul__(self, x):
        """Multiplies two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(self, x, "*")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(self, ConstantLoss(x), "*")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __rmul__(self, x):
        """Multiplies two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(x, self, "*")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(ConstantLoss(x), self, "*")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __sub__(self, x):
        """Subtracts two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(self, x, "-")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(self, ConstantLoss(x), "-")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __rsub__(self, x):
        """Subtracts two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(x, self, "-")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(ConstantLoss(x), self, "-")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __truediv__(self, x):
        """Divides two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(self, x, "/")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(self, ConstantLoss(x), "/")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __rtruediv__(self, x):
        """Divides two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(x, self, "/")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(ConstantLoss(x), self, "/")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")


class _ComposedLoss(LossBase):
    """Special class for composed loss functions

    Class for addition, multiplication, subtraction, and division of loss functions.

    Args:
        l1 (LossBase): First loss function
        l2 (LossBase): Second loss function
        composition (str): Composition of the loss functions ("+", "-", "*", "/")

    """

    def __init__(self, l1: LossBase, l2: LossBase, composition: str = "+"):
        super().__init__()
        self._l1 = l1
        self._l2 = l2
        self._composition = composition
        self._opt_param_op = self._l1._opt_param_op or self._l2._opt_param_op
        self._l1.set_opt_param_op(self._opt_param_op)
        self._l2.set_opt_param_op(self._opt_param_op)

    def set_opt_param_op(self, opt_param_op: bool = True):
        """Sets the `opt_param_op` flag.

        Args:
            opt_param_op (bool): True, if operator has trainable parameters
        """
        self._opt_param_op = opt_param_op
        self._l1.set_opt_param_op(opt_param_op)
        self._l2.set_opt_param_op(opt_param_op)

    @property
    def loss_variance_available(self) -> bool:
        if self._composition in ("*", "/"):
            return False
        else:
            return self._l1.loss_variance_available and self._l2.loss_variance_available

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for composed loss calculation."""
        return tuple(set(self._l1.loss_args_tuple + self._l2.loss_args_tuple))

    @property
    def variance_args_tuple(self) -> tuple:
        """Returns evaluation tuple for composed variance calculation."""
        if self._composition in ("*", "/"):
            raise ValueError("Variance not available for composition: ", self._composition)
        else:
            return tuple(set(self._l1.variance_args_tuple + self._l2.variance_args_tuple))

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for composed gradient calculation."""
        return tuple(set(self._l1.gradient_args_tuple + self._l2.gradient_args_tuple))

    def value(self, value_dict: dict, **kwargs) -> float:
        """Calculates and returns the composed loss value.

        Args:
            value_dict (dict): Dictionary with values for the evaluation of the loss function

        Returns:
            float: Composed loss value
        """

        value_l1 = self._l1.value(value_dict, **kwargs)
        value_l2 = self._l2.value(value_dict, **kwargs)

        if self._composition == "*":
            return value_l1 * value_l2
        elif self._composition == "/":
            return value_l1 / value_l2
        elif self._composition == "+":
            return value_l1 + value_l2
        elif self._composition == "-":
            return value_l1 - value_l2
        else:
            raise ValueError("Unknown composition: ", self._composition)

    def variance(self, value_dict: dict, **kwargs) -> float:
        """Calculates and returns the composed variance value.

        Args:
            value_dict (dict): Dictionary with values for the evaluation of the loss function

        Returns:
            float: Composed variance value
        """

        if self._composition in ("*", "/"):
            raise ValueError("Variance not available for composition: ", self._composition)

        var_l1 = self._l1.variance(value_dict, **kwargs)
        var_l2 = self._l2.variance(value_dict, **kwargs)

        if self._composition == "+":
            return var_l1 + var_l2
        elif self._composition == "-":
            return var_l1 + var_l2
        else:
            raise ValueError("Unknown composition: ", self._composition)

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Calculates and returns the gradient of the composed loss.

        Args:
            value_dict (dict): Dictionary with values for the evaluation of the
                loss function gradient

        Returns:
            Union[np.ndarray, tuple[np.ndarray, np.ndarray]]: Gradient of the composed
                loss function

        """

        grad_l1 = self._l1.gradient(value_dict, **kwargs)
        grad_l2 = self._l2.gradient(value_dict, **kwargs)
        if self._composition in ("*", "/"):
            value_l1 = self._l1.value(value_dict, **kwargs)
            value_l2 = self._l2.value(value_dict, **kwargs)

        if isinstance(grad_l1, tuple) and isinstance(grad_l2, tuple):
            if self._composition == "*":
                # (f*g)' = f'*g + f*g'
                return tuple(
                    [
                        np.add(grad_l1[i] * value_l2, grad_l2[i] * value_l1)
                        for i in range(len(grad_l1))
                    ]
                )
            elif self._composition == "/":
                # (f/g)' = (f'*g - f*g')/g^2
                return tuple(
                    [
                        np.subtract(grad_l1[i] / value_l2, value_l1 / value_l2 * grad_l2[i])
                        for i in range(len(grad_l1))
                    ]
                )
            elif self._composition == "+":
                return tuple([np.add(grad_l1[i], grad_l2[i]) for i in range(len(grad_l1))])
            elif self._composition == "-":
                return tuple([np.subtract(grad_l1[i], grad_l2[i]) for i in range(len(grad_l1))])
            else:
                raise ValueError("Unknown composition: ", self._composition)

        elif not isinstance(grad_l1, tuple) and not isinstance(grad_l2, tuple):
            if self._composition == "*":
                # (f*g)' = f'*g + f*g'
                return np.add(grad_l1 * value_l2, grad_l2 * value_l1)
            elif self._composition == "/":
                # (f/g)' = (f'*g - f*g')/g^2
                return np.subtract(grad_l1 / value_l2, value_l1 / value_l2 * grad_l2)
            elif self._composition == "+":
                return np.add(grad_l1, grad_l2)
            elif self._composition == "-":
                return np.subtract(grad_l1, grad_l2)
            else:
                raise ValueError("Unknown composition: ", self._composition)
        else:
            raise ValueError("Gradient output structure types do not match!")


class ConstantLoss(LossBase):
    """Class for constant or independent loss functions.

    Args:
        value (Union[int, float, Callable[[int],float]]): Constant value or function depending
            on the iterations returning a constant value.
    """

    def __init__(self, value: Union[int, float, Callable[[int], float]] = 0.0):
        super().__init__()
        if callable(value):
            self._value = value
        else:
            self._value = float(value)

    @property
    def loss_variance_available(self) -> bool:
        """Returns True if the loss function has a variance function."""
        return True

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns empty evaluation tuple for loss calculation."""
        return tuple()

    @property
    def variance_args_tuple(self) -> tuple:
        """Returns empty evaluation tuple for variance calculation."""
        return tuple()

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns empty evaluation tuple for gradient calculation."""
        return tuple()

    def value(self, value_dict: dict, **kwargs) -> float:
        """Returns constant or iteration dependent loss value

        Args:
            value_dict (dict): Contains calculated values of the model
            iteration (int): iteration number, if value is a callable function
        """
        if callable(self._value):
            if "iteration" not in kwargs:
                raise AttributeError("If value is callable, iteration is required.")
            return self._value(kwargs["iteration"])
        return self._value

    def variance(self, value_dict: dict, **kwargs) -> float:
        """Returns zero variance of the constant loss function."""
        return 0.0

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Returns zero gradient value

        Args:
            value_dict (dict): Contains calculated values of the model
        """
        dp = np.zeros(value_dict["param"].shape)
        dop = np.zeros(value_dict["param_op"].shape)
        if self._opt_param_op:
            return dp, dop
        return dp


class SquaredLoss(LossBase):
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
            weights (np.ndarray): Weight for each data point, if None all data points count the same

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
            weights (np.ndarray): Weight for each data point, if None all data points count the same

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
           \sum_j \sum_i w_i \left(f\left(x_i\right)-f_ref\left(x_i\right)\right) \frac{\partial f(x_i)}{\partial p_j}

        Args:
            value_dict (dict): Contains calculated values of the model
            ground_truth (np.ndarray): The true values :math:`f_ref\left(x_i\right)`
            weights (np.ndarray): Weight for each data point, if None all data points count the same
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


class VarianceLoss(LossBase):
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


class ParameterRegularizationLoss(LossBase):
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
        parameter_operator_list: Union[list, None] = [],
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
