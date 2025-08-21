"""Negative log likelihood loss function"""

import numpy as np
import sympy as sp

from .kernel_loss_base import KernelLossBase


class ODELoss(KernelLossBase):
    r"""
    ODE loss function for Quantum Kernels. It uses the same style as the QNN ODE loss function.

    This class implements the ODE loss function for Quantum Kernels. The ODE loss function is
    defined as the sum of the squared residuals of the ODE functional and the initial conditions.

    Implements an ODE Loss based on the mixed model regression algorithm of Ref. [1].

    Args:
        ODE_functional (sympy.Expr): Functional representation of the ODE (Homogeneous diferential
            equation). Must be a sympy expression and ``symbols_involved_in_ODE`` must be provided.
        symbols_involved_in_ODE (list): List of sympy symbols involved in the ODE functional.
            The list must be ordered as follows: ``[x, f, dfdx]`` where each element is a sympy
            symbol corresponding to the independent variable (``x``), the dependent variable
            (``f``), and the first derivative of the dependent variable (``dxfx``), respectively.
            There are no requirements for the symbols beyond the correct order, for example,
            ``[t, y, dydt]``.
        initial_values (np.ndarray): Initial values of the ODE. The length of the array
            must match the order of the ODE.
        boundary_handling (str): Method for handling the boundary conditions. Currently, only
            ``'pinned'``,

                * ``'pinned'``:  An extra term is added to the loss function to enforce the initial
                  values of the ODE. This term is pinned by the ``eta`` parameter. The
                  loss function is given by: :math:`L = \sum_{i=0}^{n} L_{\alpha_i}\left( \dot{f},
                  f, x  \right) + \eta \cdot (f(x_0) - f_0)^2`,
                  with :math:`f(x) = \sum \alpha_i k(x_i, x)`.

        eta (float): Weight for the initial values of the ODE in the loss function for the "pinned"
            boundary handling method.

        **Example**

        1. Implements a loss function for the ODE :math:`\cos(t) y + \frac{dy(t)}{dt} = 0` with
        initial value :math:`y(0) = 0.1`.

        .. code-block::
            t, y, dydt, = sp.symbols("t y dydt")
            eq = sp.cos(t)*y + dydt
            initial_values = [0.1]

            loss_ODE = ODELoss(
                eq,
                symbols_involved_in_ODE=[t, y, dydt],
                initial_values=initial_values,
                boundary_handling="pinned",
            )

        2. Implements a loss function for the ODE :math:`\left(df(x)/dx\right) - cos(f(x)) = 0`
        with initial values :math:`f(0) = 0.`.

        .. code-block::

            x, f, dfdx = sp.symbols("x f dfdx")
            eq = dfdx - sp.cos(f)
            initial_values = [0]

            loss_ODE = ODELoss(
                eq,
                symbols_involved_in_ODE=[x, f, dfdx],
                initial_values=initial_values,
                boundary_handling="pinned",
                eta=1.2,
            )

        References
        ----------
        [1]: A. Paine et al., "Quantum kernel methods for solving regression problems and differential equations", Phys. Rev. A 107, 032428

    Methods:
    --------
    """

    def __init__(
        self,
        ODE_functional=None,
        symbols_involved_in_ODE=None,
        initial_values: np.ndarray = None,
        eta=np.float64(1.0),
        boundary_handling="pinned",
    ):
        super().__init__()
        self._verify_size_of_ivp_with_order_of_ODE(initial_values, symbols_involved_in_ODE)
        self.order_of_ODE = (
            len(symbols_involved_in_ODE) - 2
        )  # symbols_involved_in_ODE = [x, f, f_, f__, ...]
        self.symbols_involved_in_ODE = symbols_involved_in_ODE
        self.ODE_functional = self._create_ODE_loss_format(ODE_functional, symbols_involved_in_ODE)
        self.initial_values = initial_values
        self.eta = eta
        self.boundary_handling = boundary_handling

    def _create_ODE_loss_format(self, ODE_functional, symbols_involved_in_ODE=None):
        """
        Given an ODE_functional in sympy format, returns a function that takes the derivatives list and
        returns the loss function.

        Args:
            ODE_functional (Union[Callable, sympy.Expr]): Functional representation of the ODE
                                                          (Homogeneous diferential equation).
                                                          This can be a callable function or a
                                                          sympy expression. If a sympy expression
                                                          is given, then, the
                                                          symbols_involved_in_ODE must be provided.
            symbols_involved_in_ODE (list): The list of symbols involved in the ODE problem. The
                                            list of symbols should be in order of differentiation,
                                            with the first element being the independent variable,
                                            i.e. [x, f, dfdx, dfdxdx]
        Returns:
            QNN_loss (function): The loss function for the QNN with input in the format of the QNN
                                 tuple derivatives
        """

        if isinstance(ODE_functional, sp.Expr):  # if ode_question isinstance of sympy equation
            if symbols_involved_in_ODE is None:
                raise ValueError(
                    "symbols_involved_in_ODE must be provided"
                    " if ODE_functional is a sympy equation"
                )  # Perhaps this can be somehow improved by list(ODE_functional.free_symbols)
            _ODE_functional = lambda f_alpha_tensor: sp.lambdify(
                symbols_involved_in_ODE, ODE_functional, "numpy"
            )(*f_alpha_tensor)
        else:
            raise ValueError("Only sympy expressions are allowed")

        return _ODE_functional

    def _verify_size_of_ivp_with_order_of_ODE(self, initial_values, symbols_involved_in_ODE):
        """
        Verifies that the length of the initial values vector matches the order of the ODE.

        Args:
            initial_values (np.ndarray): Initial values of the ODE
            order_of_ODE (int): Order of the ODE
        """
        order_of_ODE = len(symbols_involved_in_ODE) - 2
        if order_of_ODE != len(initial_values):
            raise ValueError(
                f"Initial values must have the same length as the order of the ODE. Order of ODE:"
                f"{len(symbols_involved_in_ODE)-2},"
                f"Length of initial values: {len(initial_values)}"
            )
        elif order_of_ODE == 2:
            print(
                "WARNING: 2nd order DEs differentiate the loss function by calculating the"
                " second derivative. This can be computationally expensive and inneficient."
            )
        elif order_of_ODE > 2:
            raise ValueError("Currently, only 1rst and 2nd order ODEs are supported")

    def compute(
        self,
        parameter_values: np.ndarray,
        data: np.ndarray,
        labels: np.ndarray,
        kernel_tensor: np.ndarray = None,  # [K, dKdx, dKdxdx] where dKdx is a np.ndarray of shape (n_samples, n_samples) and dKdxdx is a np.ndarray of shape (n_samples, n_samples)
    ) -> float:
        """
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
            labels (np.ndarray): The labels of the training data.
            kernel_tensor (array): A tensor containing the kernel matrix and its derivatives. The tensor contains the kernel matrix,  the first derivative of the kernel matrix, and the second derivative of the kernel matrix. The shapes of each element in the array are (n_samples, n_samples).

        Returns:
            float: The loss function value.

        """

        def f_alpha_order(alpha_, kernel_tensor, order):
            """Calculates the ansatz f_alpha. Order correspond to the number of times the ODE is differentiated.

            For order = 0, the ansatz is :math:`f_{\vec{\alpha}} = \alpha_0 + \sum_{i=1}^{n} \alpha_i k(x_i, x)`.
            For order = 1, the ansatz is :math:`\dot f_{\vec{\alpha}} = \sum_{i=1}^{n} \alpha_i \dot k(x_i, x)`.

            Args:
                alpha_ (np.ndarray): The vector of alphas, of shape (len(x_span)+1, 1).
                kernel_tensor (tuple): A tuple containing kernel objects for f_alpha_0 and f_alpha_1.
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

        f_alpha_tensor = np.array(
            [
                f_alpha_order(parameter_values, kernel_tensor, i)
                for i in range(self.order_of_ODE + 1)
            ]
        )
        sum1 = np.sum((self.ODE_functional([data, *f_alpha_tensor]) ** 2))  # Functional
        sum2 = np.sum(
            (f_alpha_tensor[:, 0][: len(self.initial_values)] - self.initial_values) ** 2
        )  # Initial condition
        L = sum2 + sum1 * self.eta

        return L
