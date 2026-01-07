"""Negative log likelihood loss function"""

from typing import Callable, Sequence, Union, Optional
import warnings
import numpy as np
import sympy as sp

from .kernel_loss_base import KernelLossBase


class ODELoss(KernelLossBase):
    r"""
    ODE loss function for Quantum Kernels.

    This class implements the ODE loss function for Quantum Kernels. The ODE loss function is
    defined as the sum of the squared residuals of the ODE functional and the initial conditions.

    Implements an ODE Loss based on the mixed model regression algorithm of Ref. [1].

    It uses the pinned method for handling the boundary conditions: An extra term is
    added to the loss function to enforce the initial values of the ODE.
    This term is pinned by the ``eta`` parameter. The loss function is given by:

    .. math::

        L = \sum_{i=0}^{n} L_{\alpha_i}\left( \dot{f}, f, x  \right) +
        \eta \cdot (f(x_0) - f_0)^2, \text{with} f(x) = \sum \alpha_i k(x_i, x)

    Args:
        ode_functional (:class:`sympy.core.expr.Expr` or callable)
            Functional representation of the ODE. If a :class:`sympy.core.expr.Expr` is passed,
            then `symbols_involved_in_ode` is required. If a callable is passed,
            `symbols_involved_in_ode` is optional (can be used for sanity checks) and
            `ode_order` must be provided if symbols are not given.
        initial_values (sequence or np.ndarray)
            Initial values for the ODE (length must equal ODE order).
        symbols_involved_in_ode (sequence of :class:`sympy.core.symbol.Symbol`, optional)
            Symbols in the order `[x, f, f_d1, f_d2, ...]`. Required for
            :class:`sympy.core.expr.Expr`.
        eta (float, default=1.0)
            Weight for the pinned initial-value penalty.
        ode_order (int, optional)
            Order of the ODE. If provided it is used to set/validate `order_of_ode`.
            When passing a :class:`sympy.core.expr.Expr` this must match
            `len(symbols_involved_in_ode)-2`.

    **Example**

    1. Implements a loss function for the ODE :math:`\cos(t) y + \frac{dy(t)}{dt} = 0` with
    initial value :math:`y(0) = 0.1`.

    .. code-block:: python

        t, y, dydt, = sp.symbols("t y dydt")
        eq = sp.cos(t)*y + dydt
        initial_values = [0.1]

        loss_ode = ODELoss(
            ode_functional=eq,
            initial_values=initial_values,
            symbols_involved_in_ode=[t, y, dydt],
        )

    2. Implements a loss function for the ODE :math:`\left(df(x)/dx\right) - cos(f(x)) = 0`
    with initial values :math:`f(0) = 0.`.

    .. code-block:: python

        x, f, dfdx = sp.symbols("x f dfdx")
        eq = dfdx - sp.cos(f)
        initial_values = [0]

        loss_ode = ODELoss(
            ode_functional=eq,
            initial_values=initial_values,
            symbols_involved_in_ode=[x, f, dfdx],
            eta=1.2,
        )

    See Also:
        squlearn.kernel.QKODE : Quantum Kernel for ODE solving.

    References
    ----------
        [1]: A. Paine et al., "Quantum kernel methods for solving regression problems and
        differential equations", Phys. Rev. A 107, 032428

    Methods:
    --------
    """

    def __init__(
        self,
        ode_functional: Union[sp.Expr, Callable],
        initial_values: Union[Sequence, np.ndarray],
        symbols_involved_in_ode: Optional[Sequence[sp.Basic]] = None,
        eta: float = 1.0,
        ode_order: Optional[int] = None,
    ):
        super().__init__()

        # normalize initial_values to a 1D numpy array
        self.initial_values = np.asarray(initial_values).ravel()
        self.eta = eta

        if isinstance(ode_functional, sp.Expr):
            if symbols_involved_in_ode is None:
                raise ValueError("symbols_involved_in_ode must be provided for sympy.Expr inputs")
            if ode_order is not None and ode_order != len(symbols_involved_in_ode) - 2:
                raise ValueError(
                    "ode_order does not match the length of symbols_involved_in_ode - 2"
                )
            self.order_of_ode = len(symbols_involved_in_ode) - 2
            self._check_order_of_ode_and_ivp()

            self.ode_functional = self._create_ode_loss_format_sympy(
                ode_functional, symbols_involved_in_ode
            )
        elif callable(ode_functional):
            if ode_order is None:
                raise ValueError("ode_order must be provided for callable inputs")
            if (
                symbols_involved_in_ode is not None
                and ode_order != len(symbols_involved_in_ode) - 2
            ):
                raise ValueError(
                    "ode_order does not match the length of symbols_involved_in_ode - 2"
                )
            self.order_of_ode = ode_order

            self._check_order_of_ode_and_ivp()
            self.ode_functional = self._create_ode_loss_format_callable(ode_functional)
        else:
            raise ValueError("ode_functional must be either a sympy.Expr or a callable")

    def _check_order_of_ode_and_ivp(self) -> None:
        """
        Check that the order of the ODE matches the length of the initial values.
        """
        if self.order_of_ode != len(self.initial_values):
            raise ValueError(
                f"Initial values must have the same length as the order of the ODE. Order of ODE:"
                f"{self.order_of_ode},"
                f"Length of initial values: {len(self.initial_values)}"
            )
        elif self.order_of_ode == 2:
            warnings.warn(
                "2nd order DEs differentiate the loss function by calculating the"
                " second derivative. This can be computationally expensive and inneficient."
            )
        elif self.order_of_ode > 2:
            raise ValueError("Currently, only 1rst and 2nd order ODEs are supported")

    def _create_ode_loss_format_sympy(
        self, ode_functional: sp.Expr, symbols_involved_in_ode: Sequence[sp.Basic]
    ) -> Callable:
        """
        Create a standardized loss-function wrapper for an ODE functional.

        Args:
            ode_functional (sp.Expr): Functional representation of the ODE
            symbols_involved_in_ode (Sequence): List of sympy symbols involved in the ODE
                functional.

        Returns:
            Callable: A callable that takes in a tensor of shape (order_of_ode+2 , n_samples, 1).
        """
        lamb_func = sp.lambdify(tuple(symbols_involved_in_ode), ode_functional, "numpy")

        def _ode_functional(f_alpha_tensor):
            # minimalistic wrapper: caller is responsible for correct ordering/shape
            return np.asarray(lamb_func(*f_alpha_tensor))

        return _ode_functional

    def _create_ode_loss_format_callable(self, ode_functional: Callable) -> Callable:
        """
        Create a standardized loss-function wrapper for an ODE functional given as a callable.

        Args:
            ode_functional (Callable): Functional representation of the ODE as a callable.

        Returns:
            Callable: A callable that takes in a tensor of shape (order_of_ode+2 , n_samples, 1).
        """

        def _ode_functional(f_alpha_tensor):
            try:
                return np.asarray(ode_functional(*f_alpha_tensor))
            except TypeError:
                return np.asarray(ode_functional(f_alpha_tensor))

        return _ode_functional

    def compute(
        self,
        parameter_values: np.ndarray,
        data: np.ndarray,
        **kwargs,
    ) -> float:
        r"""
        Calculates the squared loss of the loss function for the ODE as

        .. math::

            \begin{align}
                \mathcal{L}_{\vec{\alpha}} [ \ddot f,  \dot f, f,  x] &= \sum_j^N
                \left(F\left( \ddot f_{\vec{\alpha}},  \dot f_{\vec{\alpha}},
                f_{\vec{\alpha}}, x\right)_j\right)^2  + \eta\left(f_{\vec{\alpha}}(0)
                - u_0\right)^2 + \eta\left(\dot f_{\vec{\alpha}}(0) - \dot u_0\right)^2
            \end{align}

        with the ansatz :math:`f_{\vec{\alpha}} = \alpha_0 + \sum_{i=1}^{n} \alpha_i k(x_i, x)`.
        Importantly, the optimized parameters act as the coefficients of the kernel matrix
        and do not directly correspond parameterized rotations in the quantum circuit.

        Args:
            parameter_values (np.ndarray): The parameters :math:`\vec{\alpha}` of the
                ansatz to be optimized.
            data (np.ndarray): The training data to be used for the kernel matrix.
            kwargs: Additional arguments for specific loss functions.

                - kernel_tensor (np.ndarray):
                    A tensor containing the kernel matrix and its derivatives. The shapes of each
                    element in the array are (n_samples, n_samples).

        Returns:
            float: The loss function value.

        """
        kernel_tensor = kwargs.get("kernel_tensor", None)
        if kernel_tensor is None:
            raise ValueError(
                "kernel_tensor must be provided as a keyword argument for ODELoss computation"
            )

        f_alpha_tensor = np.array(
            [
                self.__f_alpha_order(parameter_values, kernel_tensor, i)
                for i in range(self.order_of_ode + 1)
            ]
        )
        sum1 = np.sum((self.ode_functional([data, *f_alpha_tensor]) ** 2))
        sum2 = np.sum(
            (f_alpha_tensor[:, 0][: len(self.initial_values)] - self.initial_values) ** 2
        )  # Initial condition
        loss_value = sum2 + sum1 * self.eta

        return loss_value

    def __f_alpha_order(self, alpha_, kernel_tensor, order):
        r"""
        Calculates the ansatz f_alpha.

        Order correspond to how often the ODE is differentiated.

        * For order = 0, the ansatz is :math:`f_{\vec{\alpha}} = \alpha_0 + \sum_{i=1}^{n}
            \alpha_i k(x_i, x)`.
        * For order = 1, the ansatz is :math:`\dot f_{\vec{\alpha}} = \sum_{i=1}^{n} \alpha_i
            \dot k(x_i, x)`.

        Args:
            alpha_ (np.ndarray): The vector of alphas, of shape (len(x_span)+1, 1).
            kernel_tensor (np.array): A tensor containing the kernel matrix and its derivatives.
                The shapes of each element in the array are (n_samples, n_samples).
            order (int): Order of the kernel.

        Returns:
            np.ndarray: The vector of f_alphas, of shape (len(x_span), 1).
        """
        alpha = alpha_[1:]
        if order == 0:
            return (
                np.dot(kernel_tensor[order], alpha).reshape(-1, 1) + alpha_[0]
            )  # shape (n_samples, 1)
        return np.dot(kernel_tensor[order], alpha).reshape(-1, 1)
