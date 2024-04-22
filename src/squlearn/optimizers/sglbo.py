from collections import deque
import numpy as np
from skopt import gp_minimize
from scipy.optimize import minimize

from .approximated_gradients import FiniteDiffGradient
from .optimizer_base import OptimizerBase, SGDMixin, default_callback, OptimizerResult


class SGLBO(OptimizerBase, SGDMixin):
    """sQUlearn's implementation of the SGLBO optimizer

    Possible options that can be set in the options dictionary are:

    * **tol** (float): Tolerance for the termination of the optimization (default: 1e-6)
    * **maxiter** (int): Maximum number of iterations per fit run (default: 100)
    * **maxiter_total** (int): Maximum number of iterations in total (default: maxiter)
    * **eps** (float): Step size for finite differences (default: 0.01)
    * **num_average** (int): Number of gradients to average (default: 1)
    * **bo_n_calls** (int): Number of iterations for the Bayesian Optimization (default: 20)
    * **bo_bounds** (List): Lower and upper bound for the search space for the Bayesian Optimization for each dimension. Each bound should be provided as a tupel (default: (0.0, 0.3))
    * **bo_bounds_fac** (float): Factor to adapt the bounds based on the gradient information (default: None)
    * **bo_n_initial_points** (int): Number of initial points for the Bayesian Optimization (default: 10)
    * **bo_x0_points** (list of lists): Initial input points (default: None)
    * **bo_aqc_optimizer** (str): Method to minimize the acquisition function. "sampling" or "lbfgs" (default: "lbfgs")
    * **bo_acq_func** (str): Acquisition function for the Bayesian Optimization (default: "EI"). Valid values are: "LCB", "EI", "PI", "gp_hedge"
    * **bo_noise** (float): Noise for noisy observations (default: "gaussian")
    * **min_surrogate** (bool): If True, the surrogate model is minimized to find the optimal step size (default: False)
    * **log_file** (str): File to log the optimization (default: None)

    Args:
        options (dict): Options for the SGLBO optimizer

    """

    def __init__(self, options: dict = None, callback=default_callback) -> None:
        super(SGDMixin, self).__init__()

        if options is None:
            options = {}

        self.tol = options.get("tol", 1e-6)
        self.maxiter = options.get("maxiter", 100)
        self.maxiter_total = options.get("maxiter_total", self.maxiter)
        self.eps = options.get("eps", 0.01)
        self.bo_n_calls = options.get("bo_n_calls", 20)
        self.bo_bounds = options.get("bo_bounds", [(0.0, 0.3)])
        self.bo_aqc_func = options.get("bo_aqc_func", "EI")
        self.bo_aqc_optimizer = options.get("bo_aqc_optimizer", "lbfgs")
        self.bo_n_initial_points = options.get("bo_n_initial_points", 10)
        self.bo_x0_points = options.get("bo_x0_points")
        self.bo_noise = options.get("bo_noise", "gaussian")
        self.bo_bounds_fac = options.get("bo_bounds_fac", None)
        self.log_file = options.get("log_file", None)
        self.min_surrogate = options.get("min_surrogate", False)
        self.num_average = options.get("num_average", 1)

        self.callback = callback
        self.options = options
        self.x = None
        self.func = None
        self.gradient_deque = deque(maxlen=self.num_average)

        if self.log_file is not None:
            f = open(self.log_file, "w")
            header = (
                f"maxiter_total: {self.maxiter_total}\n"
                f"bo_n_calls: {self.bo_n_calls}\n"
                f"bo_bounds: {self.bo_bounds}\n"
                f"bo_aqc_func: {self.bo_aqc_func}\n"
                f"bo_aqc_optimizer: {self.bo_aqc_optimizer}\n"
                f"bo_n_initial_points: {self.bo_n_initial_points}\n"
                f"bo_x0_points: {self.bo_x0_points}\n"
            )
            output = " %9s  %13s  %13s  %13s \n" % ("Iteration", "f(x)", "Gradient", "Step")
            f.write(header)
            f.write(output)
            f.close()

    def minimize(
        self,
        fun: callable,
        x0: np.ndarray,
        grad: callable = None,
        bounds=None,
    ) -> OptimizerResult:
        """
        Function to minimize a given function using the SGLBO optimizer.

        Args:
            fun (callable): Function to minimize.
            x0 (np.ndarray): Initial guess.
            grad (callable): Gradient of the function to minimize.
            bounds (sequence): Bounds for the parameters.

        Returns:
            Result of the optimization in class: `OptimizerResult` format.
        """

        self.func = fun

        # set-up number of iterations of the current run (needed for restarts)
        if self.maxiter != self.maxiter_total:
            maxiter = self.iteration + self.maxiter
        else:
            maxiter = self.maxiter_total

        self.x = x0

        if grad is None:
            grad = FiniteDiffGradient(fun, eps=self.eps).gradient

        while self.iteration < maxiter:
            # calculate the gradient
            fval = fun(self.x)
            self.gradient_deque.append(grad(self.x))
            gradient = np.average(self.gradient_deque, axis=0)

            # adapt bounds and x0 based on the gradient
            if (
                self.bo_bounds is not None
                and self.bo_x0_points is not None
                and self.bo_bounds_fac is not None
            ):
                self.bo_bounds, self.bo_x0_points = self.__adapt_bounds(
                    self.bo_bounds, self.bo_x0_points, gradient
                )

            x_updated = self.step(x=self.x, grad=gradient)

            # check termination
            if np.linalg.norm(self.x - x_updated) < self.tol:
                break

            if self.log_file is not None:
                self._log(fval, gradient, np.linalg.norm(self.x - x_updated))

            self.x = x_updated

        result = OptimizerResult()
        result.x = self.x
        result.fun = fun(self.x)
        result.nit = self.iteration

        return result

    def _get_update(self, grad: np.ndarray) -> np.ndarray:
        """Function that returns the update for a given gradient.

        Args:
            grad (np.ndarray): Gradient of the objective function.

        Returns:
            Update for the current iteration (np.ndarray).

        """
        optimal_step_size = self.__optimal_step_size(self.func, self.x, grad)
        update = optimal_step_size * grad
        return -update

    def __optimal_step_size(
        self, func: callable, start_point: np.ndarray, gradient: np.ndarray
    ) -> np.ndarray:
        """Function to estimate the optimal step size in one dimension using Bayesian Optimization.
        Args:
            func (callable): Function to minimize.
            start_point (np.ndarray): Current point in the search space.
            gradient (np.ndarray): Gradient of the objective function.

        Returns:
            Optimal step size in one dimension.
        """

        # cost function to optimize the step size in one dimension
        def step_size_cost(x):
            updated_point = start_point.copy()
            updated_point = updated_point - x * gradient

            return func(updated_point)

        # bayesian optimization to estimate the step size in one dimension
        res = gp_minimize(
            step_size_cost,
            self.bo_bounds,
            n_calls=self.bo_n_calls,
            acq_func=self.bo_aqc_func,
            acq_optimizer=self.bo_aqc_optimizer,
            x0=self.bo_x0_points,
            n_jobs=-1,
            random_state=0,
            noise=self.bo_noise,
            n_initial_points=self.bo_n_initial_points,
        )

        # minimize the surrogate model to find the optimal step size
        if self.min_surrogate:

            def func_surrogate(x):
                reg = res.models[-1]
                x = res.space.transform(x.reshape(1, -1))
                return reg.predict(x.reshape(1, -1))[0]

            res_surr = minimize(
                func_surrogate, x0=res.x[0], method="Nelder-Mead", tol=1e-6, bounds=self.bo_bounds
            )
            x_val = res_surr.x
        else:
            x_val = res.x

        return x_val

    def __adapt_bounds(
        self, current_bounds: list, current_x0: list, gradient: np.ndarray
    ) -> tuple:
        """
        Function to adapt the bounds and initial points for gp_minimize based on the gradient information.

        Args:
            current_bounds (List): Current bounds for the search space.
            current_x0 (List): Current initial points.
            gradient (np.ndarray): Gradient of the objective function.

        Returns:
            Tuple: Updated bounds for the search space and initial points.
        """

        # Compute the magnitude of the gradient
        grad_magnitude = np.linalg.norm(gradient)

        # Update the bounds based on the gradient magnitude
        updated_bounds = []
        for bound in current_bounds:
            lower = max(bound[0] - self.bo_bounds_fac * grad_magnitude, 0.0)
            upper = bound[1] + self.bo_bounds_fac * grad_magnitude
            updated_bounds.append((lower, upper))

        # Update the initial points based on the updated bounds and maintaining distribution
        updated_x0 = []

        # Distribute the x0 points evenly within the updated bounds
        for i in range(len(current_x0)):
            t = i / len(current_x0)  # t goes from 0 to 1
            updated_x0_point = [lower + t * (upper - lower) for (lower, upper) in updated_bounds]
            updated_x0.append(updated_x0_point)

        return updated_bounds, updated_x0

    def _log(self, fval, gradient, dx):
        """Function for creating a log entry of the optimization."""
        if self.log_file is not None:
            f = open(self.log_file, "a")
            if fval is not None:
                output = " %9d  %13.7f  %13.7f  %13.7f  \n" % (
                    self.iteration,
                    fval,
                    np.linalg.norm(gradient),
                    dx,
                )
            else:
                output = " %9d  %13.7f  %13.7f  \n" % (
                    self.iteration,
                    np.linalg.norm(gradient),
                    dx,
                )
            f.write(output)
            f.close()

    def reset(self):
        """Resets the optimizer to its initial state."""
        self.gradient_deque = deque(maxlen=self.num_average)
        self.x = None
        self.func = None
        self.iteration = 0

    def _update_lr(self) -> None:
        pass
