""" Various optimization methods that are implemented via wrappers
"""

import qiskit.algorithms.optimizers as qiskit_optimizers
from scipy.optimize import minimize

from .optimizer_base import OptimizerBase, OptimizerResult


class SLSQP(OptimizerBase):
    """Wrapper class for scpiy's SLSQP implementation."""

    def __init__(self, options: dict = None):
        if options is None:
            options = {}

        self.tol = options.get("tol", 1e-6)
        if "tol" in options:
            options.pop("tol")
        self.options = options

    def minimize(self, fun, x0, grad=None, bounds=None) -> OptimizerResult:
        scipy_result = minimize(
            fun, jac=grad, x0=x0, method="SLSQP", options=self.options, bounds=bounds, tol=self.tol
        )
        result = OptimizerResult()
        result.x = scipy_result.x
        result.nit = scipy_result.nit
        result.fun = scipy_result.fun
        return result


class LBFGSB(OptimizerBase):
    """Wrapper class for scpiy's L-BFGS-B implementation."""

    def __init__(self, options: dict = None):
        if options is None:
            options = {}

        self.tol = options.get("tol", 1e-6)
        if "tol" in options:
            options.pop("tol")
        self.options = options

    def minimize(self, fun, x0, grad=None, bounds=None) -> OptimizerResult:
        scipy_result = minimize(
            fun,
            jac=grad,
            x0=x0,
            method="L-BFGS-B",
            options=self.options,
            bounds=bounds,
            tol=self.tol,
        )
        result = OptimizerResult()
        result.x = scipy_result.x
        result.nit = scipy_result.nit
        result.fun = scipy_result.fun
        return result


class SPSA(OptimizerBase):
    """Wrapper class for Qiskit's SPSA implementation."""

    def __init__(self, options: dict = None):
        if options is None:
            options = {}

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

    def minimize(self, fun, x0, grad=None, bounds=None) -> OptimizerResult:
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
        )

        result_qiskit = spsa.minimize(fun=fun, x0=x0, jac=grad, bounds=bounds)

        result = OptimizerResult()
        result.x = result_qiskit.x
        result.nit = result_qiskit.nit
        result.fun = result_qiskit.fun
        return result
