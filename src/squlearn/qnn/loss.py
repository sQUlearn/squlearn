"""Loss function implementations for QNNs."""

import abc
from typing import Union
import numpy as np
import sympy as sp
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
           2\sum_j \sum_i w_i \left(f\left(x_i\right)-f_ref\left(x_i\right)\right) \frac{\partial f(x_i)}{\partial p_j}

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


class ODELoss(LossBase):
    """Squared loss for regression of Ordinary Differential Equations (ODEs).
    Implements an ODE Loss based on Ref. [1].

    Args:
        ODE_functional (Union[Callable, sympy.Expr]): Functional representation of the ODE (Homogeneous diferential equation). This can be a callable function or a sympy expression.
            If a callable function is given, then, a callable function for the gradient (ODE_functional_gradient) of the homogenous differential equation calculation must be provided.
            If a sympy expression is given, then, the symbols_involved_in_ODE must be provided.
        ODE_functional_gradient (Callable): Only necessary if ODE_functional is a callable function. A callable function that returns the gradient with regards to the partial derivatives of the function, the first derivative, and the second derivative: (dF/df, dF/df_, dF/df__).
        initial_vec (np.ndarray): Initial values of the ODE. If only one value is given, then it is a 1rst order ODE. If two values are given, then it is a 2nd order ODE.
        boundary_handling (str): Method for handling the boundary conditions. Options are "pinned", and "floating".
            - "pinned":   An extra term is added to the loss function to enforce the initial values of the ODE. This term is pinned by the eta parameter. The lost function is given by: L = \sum_i=0^n_samples L_theta_i + eta*(f(x_0) - f_0)^2
            - "floating": The initial value of the ivp is fixed by definition and the loss function is calculated without the corresponding ivp term. The QNN is allowed to "float" around the initial values.  L = \sum_i=1^n_samples L_theta_i
        eta (float): Weight for the initial values of the ODE in the loss function for the "pinned" boundary handling method.
        symbols_involved_in_ODE (list): List of sympy symbols involved in the ODE functional.  The list must be ordered as follows: [x, f, f_] where x is the independent variable, f is the function and f_ is the first derivative.

    References
    ----------
    [1]: O. Kyriienko et al., "Solving nonlinear differential equations with differentiable quantum circuits",
    `arXiv:2011.10395 (2021). <https://arxiv.org/pdf/2011.10395>`_
    """

    def __init__(
        self,
        ODE_functional=None,
        ODE_functional_gradient=None,
        initial_vec: np.ndarray = None,
        eta=np.float64(1.0),
        boundary_handling="pinned",
        symbols_involved_in_ODE=None,
    ):
        super().__init__()
        self._ODE_functional = self.create_QNN_ODE_loss_format(
            ODE_functional, symbols_involved_in_ODE
        )  # F[x, f, f_, f__] returns the value of the ODE functional shape: (n_samples, n_outputs)
        self._ODE_functional_gradient_dp = self.create_QNN_ODE_gradient_format(
            ODE_functional_gradient, "dfdp", ODE_functional, symbols_involved_in_ODE
        )  # (dF/df, dF/df_, dF/df__) returns the value of the ODE functional shape: (order_of_ODE, n_samples, num_param)
        self._ODE_functional_gradient_dop = self.create_QNN_ODE_gradient_format(
            ODE_functional_gradient, "dfdop", ODE_functional, symbols_involved_in_ODE
        )  # (dF/df, dF/df_, dF/df__) returns the value of the ODE functional shape: (order_of_ODE+1, n_samples, num_param_op)
        self.initial_vec = initial_vec
        self.order_of_ODE = len(initial_vec)
        self.eta = eta
        self.boundary_handling = boundary_handling

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for the squared loss calculation."""
        if self.order_of_ODE == 1:  # if only one initial value is given, we have a 1rst order ODE
            return ("f", "dfdx")
        elif self.order_of_ODE == 2:
            print(
                "WARNING: 2nd order DEs differentiate the QNN loss function by calculating the second derivative. This can be computationally expensive and inneficient. An alternative is to set-up coupled 1rst order DEs (currently not implemented)"
            )
            return ("f", "dfdx", "dfdxdx")

    def get_true_solution(self) -> np.ndarray:
        return self.true_solution

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for the squared loss gradient calculation."""
        if self._opt_param_op:
            if (
                self.order_of_ODE == 1
            ):  # if only one initial value is given, we have a 1rst order ODE
                return ("f", "dfdx", "dfdp", "dfdxdp", "dfdop", "dfdopdx")
            elif self.order_of_ODE == 2:
                return (
                    "f",
                    "dfdx",
                    "dfdxdx",
                    "dfdp",
                    "dfdxdp",
                    "dfdxdxdp",
                    "dfdop",
                    "dfdopdx",
                    "dfdopdxdx",
                )

        if self.order_of_ODE == 1:  # if only one initial value is given, we have a 1rst order ODE
            return ("f", "dfdx", "dfdp", "dfdxdp")
        elif self.order_of_ODE == 2:
            return ("f", "dfdx", "dfdxdx", "dfdp", "dfdxdp", "dfdxdxdp")

    def derivatives_in_array_format(self, loss_values):
        """
        Given a dictionary of loss_values, returns the values in the format of the QNN tuple derivatives

        Args:
            loss_values (dict): Contains calculated values of the model
        Returns:
            x (np.ndarray): The input values
            f (np.ndarray): The output values
            dfdx (np.ndarray): The first derivative values
            dfdxdx (np.ndarray): The second derivative values

        """
        if self.order_of_ODE == 1:  # if only one initial value is given, we have a 1rst order ODE
            return (
                loss_values["x"],
                loss_values["f"],
                loss_values["dfdx"][:, 0],
            )
        elif self.order_of_ODE == 2:  # if two initial value are given, we have a 2nd order ODE
            return (
                loss_values["x"],
                loss_values["f"],
                loss_values["dfdx"][:, 0],
                loss_values["dfdxdx"][:, 0, 0],
            )

    def _ansatz_to_floating_boundary_ansatz(
        self, value_dict_floating: dict, gradient_calculation=True, **kwargs
    ) -> dict:
        """
        Converts the value_dict_floating to a floating boundary ansatz by shifting the output values by a bias term that includes the initial values of the ODE.

        If 1rst order ODE: f(x) = f(x) - f_b, with f_b = f(x_0) - f_0 and f'(x) = f'(x) - f'(x_0)
        If 2nd order ODE:  f(x) = f(x) - f_b, with f_b = f(x_0) - f_0 and f'(x) = f'(x) - f_b' and f''(x) = f''(x) - f''(x_0)

        Args:
            value_dict (dict): Contains calculated values of the model
            gradient_calculation (bool): True if the gradient is calculated

        Returns:
            value_dict_floating (dict): Contains the values of the model with the initial values set to the initial values of the ODE


        """
        f_bias = value_dict_floating["f"][0] - self.initial_vec[0]  # f_b = f(x_0) - f_0
        value_dict_floating["f"] -= f_bias  # f(x) = f(x) - f_b

        if self.order_of_ODE == 2:  # if only one initial value is given, we have a 1rst order ODE
            f_bias = value_dict_floating["dfdx"][0] - self.initial_vec[1]  # f_b = f'(x_0) - f_0'
            value_dict_floating["dfdx"] -= f_bias  # f'(x) = f'(x) - f_b

        if gradient_calculation:
            df_biasdp = value_dict_floating["dfdp"][0]  # df_b/dp = df(x_0)/dp
            value_dict_floating["dfdp"] -= df_biasdp  # df(x)/dp = df(x)/dp - df_b/dp
            if self._opt_param_op:
                df_biasdop = value_dict_floating["dfdop"][0]
                value_dict_floating["dfdop"] -= df_biasdop

            if self.order_of_ODE == 2:  # if two initial value are given, we have a 2nd order ODE
                df_biasdxdp = value_dict_floating["dfdxdp"][0]  # df_b/dp = df(x_0)/dp
                value_dict_floating["dfdxdp"] -= df_biasdxdp  # df(x)/dp = df(x)/dp - df_b/dp
                if self._opt_param_op:
                    df_biasdxdop = value_dict_floating["dfdxdop"][0]
                    value_dict_floating["dfdxdop"] -= df_biasdxdop
        return value_dict_floating

    def value(self, value_dict: dict, **kwargs) -> float:
        r"""
        Calculates the squared loss of the loss function for the ODE as

        .. math::
            \begin{align}
                \mathcal{L}_{\vec{\theta}} [ \ddot f,  \dot f, f,  x] &= \sum_j^N \left(F\left( \ddot f_{\vec{\theta}},  \dot f_{\vec{\theta}}, f_{\vec{\theta}}, x\right)_j\right)^2  + \eta\left(f_{\vec{\theta}}(0) - u_0\right)^2 + \eta\left(\dot f_{\vec{\theta}}(0) - \dot u_0\right)^2
            \end{align}

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

        multiple_output = "multiple_output" in kwargs and kwargs["multiple_output"]

        functional_loss, initial_value_loss_f, initial_value_loss_df = 0, 0, 0

        if multiple_output:
            raise NotImplementedError("Coupled ODEs and/or PDE are not implemented yet.")
        else:
            if self.boundary_handling == "pinned":
                functional_loss = np.sum(
                    np.multiply(
                        np.square(self._ODE_functional(value_dict) - ground_truth), weights
                    )
                )  # L_theta = sum_i w_i (F(x_i, f_i, f_i', f_i'') - 0)^2, shape (n_samples, 1)

                initial_value_loss_f = self.eta * (
                    np.square(value_dict["f"][0] - self.initial_vec[0])
                )  # L_theta =  eta * (f(x_i) - f_0)^2 #Pinned boundary condition
                if self.order_of_ODE == 2:
                    initial_value_loss_df = self.eta * (
                        np.square(value_dict["dfdx"][0] - self.initial_vec[1])
                    )  # L_theta =  eta * (f'(x_i) - f_0')^2
                else:
                    pass
            elif self.boundary_handling == "floating":
                value_dict = self._ansatz_to_floating_boundary_ansatz(
                    value_dict, gradient_calculation=False
                )
                functional_loss = np.sum(
                    np.multiply(
                        np.square(self._ODE_functional(value_dict) - ground_truth), weights
                    )
                )  # L_theta = sum_i w_i (F(x_i, f_i, f_i', f_i'') - 0)^2, shape (n_samples, n_outputs)
            elif self.boundary_handling == "optimized":
                raise NotImplementedError("Optimized boundary handling not implemented yet.")
            # print(functional_loss + initial_value_loss_f + initial_value_loss_df)
            return functional_loss + initial_value_loss_f + initial_value_loss_df

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        r"""Returns the gradient of the squared loss.

        Calculates the gradient of the squared loss between the values in value_dict
        and ground_truth as

        .. math::
            \begin{align}
                \pdv{\mathcal{L}_{\vec{\theta}}}{\theta_i} &=  \sum_j^N 2 \left(F[ \ddot f_{\vec{\theta}},  \dot f_{\vec{\theta}}, f_{\vec{\theta}}, x]_j\right) \pdv{}{\theta_i} \left(F[ \ddot f_{\vec{\theta}},  \dot f_{\vec{\theta}}, f_{\vec{\theta}}, x]_j \right)  + 2 \eta(f_{\vec{\theta}}(0)-u_0) \eval{\pdv{f_{\vec{\theta}}(x)}{\theta_i}}_{x=0} + 2 \eta(\dot f_{\vec{\theta}}(0)- \dot u_0) \eval{\pdv{\dot f_{\vec{\theta}}(x)}{\theta_i}}_{x=0} \\
                &= \sum_j^N 2 \left(F[ \ddot f_{\vec{\theta}},  \dot f_{\vec{\theta}}, f_{\vec{\theta}}, x]_j\right) \left( \pdv{F_j}{f_{\vec{\theta}}}\pdv{f_{\vec{\theta}}}{\theta_i}
                +  \pdv{F_j}{\dot f_{\vec{\theta}}}\pdv{\dot f_{\vec{\theta}}}{\theta_i} +  \pdv{F_j}{\ddot f_{\vec{\theta}}}\pdv{\ddot f_{\vec{\theta}}}{\theta_i}\right) 
                + 2 \eta(f_{\vec{\theta}}(0)-u_0) \eval{\pdv{f_{\vec{\theta}}(x)}{\theta_i}}_{x=0} + 2 \eta(\dot f_{\vec{\theta}}(0)- \dot u_0) \eval{\pdv{\dot f_{\vec{\theta}}(x)}{\theta_i}}_{x=0} 
            \end{align}


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

        weighted_diff = np.multiply(
            (self._ODE_functional(value_dict) - ground_truth), weights
        )  # shape: (n_samples, 1)

        if value_dict["dfdp"].shape[0] == 0:
            d_p = np.array([])
        else:
            if multiple_output:
                raise NotImplementedError("Coupled ODEs and/or PDE are not implemented yet.")
            else:
                # value_dict["dfdp"] shape: (n_samples, n_params)
                # value_dict["dfdxdp"] shape: (n_samples, 1, n_params)
                d_p = np.zeros(value_dict["dfdp"].shape[1])  # shape: (n_params)
                if self.boundary_handling == "pinned":
                    d_p += (
                        2.0
                        * self.eta
                        * (value_dict["f"][0] - self.initial_vec[0])
                        * value_dict["dfdp"][0, :]
                    )  # shape: (n_params)
                    if self.order_of_ODE == 2:
                        d_p += (
                            2.0
                            * self.eta
                            * np.sum(value_dict["dfdx"][0] - self.initial_vec[1])
                            * value_dict["dfdxdp"][0, 0, :]
                        )  # shape: (n_params)

                elif self.boundary_handling == "floating":
                    value_dict = self._ansatz_to_floating_boundary_ansatz(
                        value_dict, gradient_calculation=True
                    )

                d_ODE_functional_dD = self._ODE_functional_gradient_dp(
                    value_dict
                )  # shape: (1+self.order_of_ODE, n_samples, n_params)

                if self.order_of_ODE == 1:
                    dfdp_like = (
                        d_ODE_functional_dD[0] * value_dict["dfdp"]
                        + d_ODE_functional_dD[1] * value_dict["dfdxdp"][:, 0, :]
                    )  # shape: (n_samples, n_params)
                else:
                    dfdp_like = (
                        d_ODE_functional_dD[0] * value_dict["dfdp"]
                        + d_ODE_functional_dD[1] * value_dict["dfdxdp"][:, 0, :]
                        + d_ODE_functional_dD[2] * value_dict["dfdxdxdp"][:, 0, 0, :]
                    )

                d_p += 2.0 * np.einsum(
                    "j,jk->k", weighted_diff, dfdp_like
                )  # shape: (n_samples, n_params) -> (n_params)

        if not self._opt_param_op:
            return d_p

        if value_dict["dfdop"].shape[0] == 0:
            d_op = np.array([])
        else:
            if multiple_output:
                raise NotImplementedError("Coupled ODEs and/or PDE are not implemented yet.")
            else:
                d_op = np.zeros(value_dict["dfdop"].shape[1])  # shape: (n_param_op)
                if self.boundary_handling == "pinned":
                    d_op += (
                        2.0
                        * self.eta
                        * (value_dict["f"][0] - self.initial_vec[0])
                        * value_dict["dfdop"][0, :]
                    )
                    if self.order_of_ODE == 2:
                        d_op += (
                            2.0
                            * self.eta
                            * np.sum(value_dict["dfdx"][0] - self.initial_vec[1])
                            * value_dict["dfdopdx"][0, 0, :]
                        )

                d_ODE_functional_dD = self._ODE_functional_gradient_dop(
                    value_dict
                )  # shape: (1+self.order_of_ODE, n_samples, n_param_op)

                if self.order_of_ODE == 1:
                    dfdop_like = (
                        d_ODE_functional_dD[0] * value_dict["dfdop"]
                        + d_ODE_functional_dD[1] * value_dict["dfdopdx"][:, 0, :]
                    )  # shape: (n_samples, n_param_op)
                else:
                    dfdop_like = (
                        d_ODE_functional_dD[0] * value_dict["dfdop"]
                        + d_ODE_functional_dD[1] * value_dict["dfdopdx"][:, 0, :]
                        + d_ODE_functional_dD[2] * value_dict["dfdopdxdx"][:, 0, 0, :]
                    )

                d_op += 2.0 * np.einsum(
                    "j,jk->k", weighted_diff, dfdop_like
                )  # shape: (n_samples, n_param_op) -> (n_param_op)

        return d_p, d_op

    def create_QNN_ODE_loss_format(self, ODE_functional, symbols_involved_in_ODE=None):
        """
        Given an ODE_functional, returns a function that takes the QNN derivatives list and returns the loss function.

        Args:
            ODE_functional (Union[Callable, sympy.Expr]): Functional representation of the ODE (Homogeneous diferential equation). This can be a callable function or a sympy expression. If a sympy expression is given, then, the symbols_involved_in_ODE must be provided.
            symbols_involved_in_ODE (list): The list of symbols involved in the ODE problem. The list of symbols should be in order of differentiation, with the first element being the independent variable, i.e. [x, f, dfdx, dfdxdx]
        Returns:
            QNN_loss (function): The loss function for the QNN with input in the format of the QNN tuple derivatives
        """

        if isinstance(ODE_functional, sp.Expr):  # if ode_question isinstance of sympy equation
            if symbols_involved_in_ODE is None:
                raise ValueError(
                    "symbols_involved_in_ODE must be provided if ODE_functional is a sympy equation"
                )  # Perhaps this can be somehow improved by list(ODE_functional.free_symbols)
            _ODE_functional = numpyfy_sympy_loss(ODE_functional, symbols_involved_in_ODE)
        else:
            _ODE_functional = ODE_functional

        def QNN_loss(value_dict):
            """
            Given the squlearn QNN derivatives dictionary, returns the loss function defined by the ODE problem.

            Args:
                value_dict (dict): Contains calculated values of the model from squlearn QNN
            Returns:
                loss (np.ndarray): The loss evaluated at the n_samples input values.  shape: (n_samples,)
            """
            return _ODE_functional(self.derivatives_in_array_format(value_dict))

        return QNN_loss

    def create_QNN_ODE_gradient_format(
        self,
        ODE_functional_gradient,
        dimension_of_gradient_with_respect_to,
        ODE_functional,
        symbols_involved_in_ODE=None,
    ):
        """
        Given an ODE_functional_gradient, returns a function that takes the QNN derivatives list and returns the gradient of the loss function.

        Args:
            ODE_functional_gradient (Union[Callable, None]): Gradient of the ODE functional. A callable function that takes the QNN derivatives list and returns the gradient of the loss function. If None, the gradient is calculated using the sympy expression of the ODE_functional
            dimension_of_gradient_with_respect_to (str): The dimension of the gradient with respect to the parameters. It can be "dfdp" or "dfdop"
        Returns:
            QNN_gradient (function): The gradient function for the QNN with input in the format of the QNN tuple derivatives
        """
        if ODE_functional_gradient is None:
            _ODE_functional_gradient = numerical_gradient_of_symbolic_equation(
                ODE_functional, symbols_involved_in_ODE
            )
        else:
            _ODE_functional_gradient = ODE_functional_gradient

        def QNN_gradient(value_dict):
            """
            Given the squlearn QNN derivatives dictionary, returns the gradient of the loss function defined by the ODE problem.

            Args:
                value_dict (dict): Contains calculated values of the model from squlearn QNN
            Returns:
                grad_envelope_list (np.ndarray): The gradient of the loss evaluated at the n_samples input values.  shape: (3, n_samples, n_params)

            """
            dF_dpartial = _ODE_functional_gradient(
                self.derivatives_in_array_format(value_dict)
            )  # [dFdf(n_samples, 1), dFdfdx(n_samples, 1)] or [1, dFdfdx(n_samples, 1)] or [dFdf(n_samples, 1), 1] if one of the derivatives returns a scalar, that is why we need to pile them up in the next step
            n_param = value_dict[dimension_of_gradient_with_respect_to].shape[1]

            grad_envelope_list = np.zeros(
                (len(dF_dpartial), value_dict["x"].shape[0], n_param)
            )  # shape (3, n, n_param)
            for i in range(len(dF_dpartial)):
                grad_envelope_list[i, :, :] = np.tile(
                    dF_dpartial[i], (n_param, 1)
                ).T  # This is necessary to broadcast the gradient to the n_param dimensions

                # EXAMPLE:  dF_dpartial = [dFdf, dFdfdx]
                # dF dpartial [array([6.77020277, 7.02029189, 7.21093346, 7.33490286]), 1]
                # --------------------
                # grad_envelope_list [[[6.77020277 6.77020277]
                # [7.02029189 7.02029189]
                # [7.21093346 7.21093346]
                # [7.33490286 7.33490286]]

                # [[1.         1.        ]
                # [1.         1.        ]
                # [1.         1.        ]
                # [1.         1.        ]]]

            return grad_envelope_list

        return QNN_gradient


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


def numpyfy_sympy_loss(sp_ode, symbols_involved_in_ODE):
    """
    Given a sympy equation, returns a function that takes the QNN derivatives in a list and returns the loss value.

    Args:
        sp_ode (sympy equation): The sympy equation of the ODE problem
        symbols_involved_in_ODE (list): The list of symbols involved in the ODE problem the list of symbols should be in order [x, f, dfdx, dfdxdx]
    Returns:
        numpy_loss (function): The loss function for the QNN with input in the format of the QNN tuple derivatives

    """

    def numpy_loss(f_alpha_tensor):
        return sp.lambdify(symbols_involved_in_ODE, sp_ode, "numpy")(*f_alpha_tensor)

    return numpy_loss


def symbolic_gradient(sp_ode, f_arguments):
    """
    Calculate the gradient of a sympy equation with respect to a given set of symbols,
    Args:

    sp_ode (sympy equation): The sp_ode to calculate the gradient of.
    f_arguments (list of sympy symbols): The variables to calculate the gradient with respect to. Assumes [f, dfdx, ...]

    Returns:
    list of sympy equations: The gradient of the sp_ode with respect to the given variables.

    """
    gradients = []
    for f_order in f_arguments:
        gradients.append(sp.diff(sp_ode, f_order))
    return gradients


def numerical_gradient_of_symbolic_equation(sp_ode, symbols_involved_in_ODE):
    """
    Calculate the gradient of a sympy equation with respect to a given set of variables,

    Args:

    equation (sympy equation): The equation to calculate the gradient of.
    symbols_involved_in_ODE (list of sympy symbols): Assumes [x, f, dfdx, ...]

    Returns:
    list of sympy equations: The gradient of the equation with respect to the given variables.

    """
    gradients = symbolic_gradient(sp_ode, symbols_involved_in_ODE[1:])

    def np_grad_out_sp(f_alpha_tensor):
        return [
            sp.lambdify(symbols_involved_in_ODE, grad_i, "numpy")(*f_alpha_tensor)
            for grad_i in gradients
        ]

    return np_grad_out_sp
