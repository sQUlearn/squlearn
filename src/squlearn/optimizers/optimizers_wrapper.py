""" Various optimization methods that are implemented via wrappers"""

import numpy as np
import qiskit_algorithms.optimizers as qiskit_optimizers
from scipy.optimize import minimize

from .optimizer_base import OptimizerBase, OptimizerResult, IterativeMixin, default_callback


class WrappedOptimizerBase(OptimizerBase, IterativeMixin):
    """Base class for wrapped optimizers.

    Overwrites the set_callback function to additionally increase the iteration counter.
    """

    def set_callback(self, callback):
        """Set the callback function with additional iteration counter increasing."""

        def callback_wrapper(*args):
            nonlocal self
            self.iteration += 1
            callback(*args)

        super().set_callback(callback_wrapper)


class SLSQP(WrappedOptimizerBase):
    """Wrapper class for scipy's SLSQP implementation.

    Args:
        options (dict): Options for the SLSQP optimizer.
                        The options are the same as for :meth:`scipy.optimize.minimize`
    """

    def __init__(self, options: dict = None, callback=default_callback):
        super().__init__()

        if options is None:
            options = {}

        self.tol = options.get("tol", 1e-6)
        if "tol" in options:
            options.pop("tol")
        self.options = options
        self.set_callback(callback)

    def minimize(
        self, fun: callable, x0: np.ndarray, grad: callable = None, bounds=None
    ) -> OptimizerResult:
        """
        Function to minimize a given function using the SLSQP optimizer. Is wrapped from scipy.

        Args:
            fun (callable): Function to minimize.
            x0 (numpy.ndarray): Initial guess.
            grad (callable): Gradient of the function to minimize.
            bounds (sequence): Bounds for the parameters.

        Returns:
            Result of the optimization in class:`OptimizerResult` format.
        """

        scipy_result = minimize(
            fun,
            jac=grad,
            x0=x0,
            method="SLSQP",
            options=self.options,
            bounds=bounds,
            tol=self.tol,
            callback=self.callback,
        )
        result = OptimizerResult()
        result.x = scipy_result.x
        result.nit = scipy_result.nit
        result.fun = scipy_result.fun
        return result


class LBFGSB(WrappedOptimizerBase):
    """Wrapper class for scipy's L-BFGS-B implementation.

    Args:
        options (dict): Options for the L-BFGS-B optimizer.
                        The options are the same as for :meth:`scipy.optimize.minimize`
    """

    def __init__(self, options: dict = None, callback=default_callback):
        super().__init__()

        if options is None:
            options = {}

        self.tol = options.get("tol", 1e-6)
        if "tol" in options:
            options.pop("tol")
        self.options = options
        self.set_callback(callback)

    def minimize(
        self, fun: callable, x0: np.ndarray, grad: callable = None, bounds=None
    ) -> OptimizerResult:
        """
        Function to minimize a given function using the L-BFGS-B optimizer. Is wrapped from scipy.

        Args:
            fun (callable): Function to minimize.
            x0 (numpy.ndarray): Initial guess.
            grad (callable): Gradient of the function to minimize.
            bounds (sequence): Bounds for the parameters.

        Returns:
            Result of the optimization in class:`OptimizerResult` format.
        """
        scipy_result = minimize(
            fun,
            jac=grad,
            x0=x0,
            method="L-BFGS-B",
            options=self.options,
            bounds=bounds,
            tol=self.tol,
            callback=self.callback,
        )
        result = OptimizerResult()
        result.x = scipy_result.x
        result.nit = scipy_result.nit
        result.fun = scipy_result.fun
        return result


class SPSA(WrappedOptimizerBase):
    """Wrapper class for Qiskit's SPSA implementation.

    Args:
        options (dict): Options for the SPSA optimizer.
                        The options are the same as for :meth:`qiskit_algorithms.optimizers.SPSA`
    """

    def __init__(self, options: dict = None, callback=default_callback):
        super().__init__()
        self.set_callback(callback)

        if options is None:
            options = {}

        self.options = options
        self.maxiter = options.get("maxiter", 100)
        self.blocking = options.get("blocking", False)
        self.allowed_increase = options.get("allowed_increase", None)
        self.trust_region = options.get("trust_region", False)
        self.learning_rate = options.get("learning_rate", None)
        self.perturbation = options.get("perturbation", None)
        self.last_avg = options.get("last_avg", 1)
        self.resamplings = options.get("resamplings", 1)
        self.perturbation_dims = options.get("perturbation_dims", None)
        self.second_order = options.get("second_order", False)
        self.regularization = options.get("regularization", None)
        self.hessian_delay = options.get("hessian_delay", 0)
        self.lse_solver = options.get("lse_solver", None)
        self.initial_hessian = options.get("initial_hessian", None)
        self.callback = callback

    def minimize(
        self, fun: callable, x0: np.ndarray, grad: callable = None, bounds=None
    ) -> OptimizerResult:
        """
        Function to minimize a given function using Qiskit's SPSA optimizer.

        Args:
            fun (callable): Function to minimize.
            x0 (numpy.ndarray): Initial guess.
            grad (callable): Gradient of the function to minimize.
            bounds (sequence): Bounds for the parameters.

        Returns:
            Result of the optimization in class:`OptimizerResult` format.
        """

        spsa = qiskit_optimizers.SPSA(
            maxiter=self.maxiter,
            blocking=self.blocking,
            allowed_increase=self.allowed_increase,
            trust_region=self.trust_region,
            learning_rate=self.learning_rate,
            perturbation=self.perturbation,
            last_avg=self.last_avg,
            resamplings=self.resamplings,
            perturbation_dims=self.perturbation_dims,
            second_order=self.second_order,
            regularization=self.regularization,
            hessian_delay=self.hessian_delay,
            lse_solver=self.lse_solver,
            initial_hessian=self.initial_hessian,
            callback=self.callback,
        )

        result_qiskit = spsa.minimize(fun=fun, x0=x0, jac=grad, bounds=bounds)

        result = OptimizerResult()
        result.x = result_qiskit.x
        result.nit = result_qiskit.nit
        result.fun = result_qiskit.fun
        return result
