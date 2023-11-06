import abc
from collections import deque
import numpy as np

from .optimizer_base import OptimizerBase, SGDMixin, OptimizerResult, default_callback
from .approximated_gradients import FiniteDiffGradient


class Adam(OptimizerBase, SGDMixin):
    """sQUlearn's implementation of the ADAM optimizer

    Possible options that can be set in the options dictionary are:

    * **tol** (float): Tolerance for the termination of the optimization (default: 1e-6)
    * **lr** (float, list, np.ndarray, callable): Learning rate. If float, the learning rate is constant.
      If list or np.ndarray, the learning rate is taken from the list or array.
      If callable, the learning rate is taken from the function. (default: 0.05)
    * **beta_1** (float): Decay rate for the first moment estimate (default: 0.9)
    * **beta_2** (float): Decay rate for the second moment estimate (default: 0.99)
    * **regularization** (float): Small value to avoid division by zero (default: 1e-8)
    * **num_average** (int): Number of gradients to average (default: 1)
    * **maxiter** (int): Maximum number of iterations per fit run (default: 100)
    * **maxiter_total** (int): Maximum number of iterations in total (default: maxiter)
    * **log_file** (str): File to log the optimization (default: None)
    * **skip_fun** (bool): If True, the function evaluation is skipped (default: False)
    * **eps** (float): Step size for finite differences (default: 0.01)

    Args:
        options (dict): Options for the ADAM optimizer.

    """

    def __init__(self, options: dict = None, callback=default_callback) -> None:  # TODO: kwargs?
        super(SGDMixin, self).__init__()  #  set-up of self.iterations=0

        if options is None:
            options = {}

        self.tol = options.get("tol", 1e-6)
        self.lr = options.get("lr", 0.05)
        self.beta_1 = options.get("beta_1", 0.9)
        self.beta_2 = options.get("beta_2", 0.99)
        self.regularization = options.get("regularization", 1e-8)
        self.num_average = options.get("num_average", 1)
        self.maxiter = options.get("maxiter", 100)
        self.maxiter_total = options.get("maxiter_total", self.maxiter)
        self.log_file = options.get("log_file", None)
        self.skip_fun = options.get("skip_fun", False)
        self.eps = options.get("eps", 0.01)

        self.callback = callback
        self.options = options

        self.gradient_deque = deque(maxlen=self.num_average)
        self.m = None
        self.v = None
        self.x = None
        self.lr_eff = 0.0
        self._update_lr()

        if self.log_file is not None:
            f = open(self.log_file, "w")
            if self.skip_fun is not None:
                output = " %9s  %12s  %12s  %12s  %12s  %12s \n" % (
                    "Iteration",
                    "f(x)",
                    "Gradient",
                    "Step",
                    "Eff. LR",
                    "LR",
                )
            else:
                output = " %9s  %12s  %12s  %12s  %12s \n" % (
                    "Iteration",
                    "Gradient",
                    "Step",
                    "Eff. LR",
                    "LR",
                )
            f.write(output)
            f.close()

    def reset(self):
        """Resets the optimizer to its initial state."""
        self.gradient_deque = deque(maxlen=self.num_average)
        self.m = None
        self.v = None
        self.x = None
        self.lr_eff = 0.0
        self.iteration = 0
        self._update_lr()

    def minimize(
        self, fun: callable, x0: np.ndarray, grad: callable = None, bounds=None
    ) -> OptimizerResult:
        """
        Function to minimize a given function using the ADAM optimizer.

        Args:
            fun (callable): Function to minimize.
            x0 (numpy.ndarray): Initial guess.
            grad (callable): Gradient of the function to minimize.
            bounds (sequence): Bounds for the parameters.

        Returns:
            Result of the optimization in class:`OptimizerResult` format.
        """

        # set-up number of iterations of the current run (needed for restarts)
        if self.maxiter != self.maxiter_total:
            maxiter = self.iteration + self.maxiter
        else:
            maxiter = self.maxiter_total

        self.x = x_updated = x0

        if grad is None:
            grad = FiniteDiffGradient(fun, eps=self.eps).gradient

        while self.iteration < maxiter:
            # calculate the function value (TODO: make it skipable)
            if self.skip_fun:
                fval = None
            else:
                fval = fun(self.x)

            # Calculate the gradient and average it over the last num_average gradients
            # (1 is default: no averaging)
            self.gradient_deque.append(grad(self.x))
            gradient = np.average(self.gradient_deque, axis=0)

            x_updated = self.step(x=self.x, grad=gradient)

            if bounds != None:
                x_updated = np.clip(x_updated, bounds[:, 0], bounds[:, 1])

            if self.log_file is not None:
                self._log(fval, gradient, np.linalg.norm(self.x - x_updated))

            if self.callback is not None:
                self.callback(self.iteration, self.x, gradient, fval)

            # check termination
            if np.linalg.norm(self.x - x_updated) < self.tol:
                break

            self.x = x_updated

        result = OptimizerResult()
        result.x = self.x
        result.fun = fun(self.x)
        result.nit = self.iteration

        return result

    def _get_update(self, grad: np.ndarray):
        """Function that returns the update for a given gradient.

        Args:
            grad (np.ndarray): Gradient of the objective function.

        Returns:
            Update for the current iteration (np.ndarray).

        """
        if self.m is None:
            self.m = np.zeros(np.shape(grad))

        if self.v is None:
            self.v = np.zeros(np.shape(grad))

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * grad * grad

        return (
            -1.0
            * self.lr_eff
            * self.m.flatten()
            / (np.sqrt(self.v.flatten()) + self.regularization)
        )

    def _update_lr(self):
        """Function that calculates the effective learning rate."""

        # Get effective learning rate
        if callable(self.lr):
            lr_val = self.lr(self.iteration)
        elif isinstance(self.lr, list) or isinstance(self.lr, np.ndarray):
            lr_val = self.lr[self.iteration]
        else:
            lr_val = self.lr

        if self.iteration <= 0:
            self.lr_eff = lr_val * np.sqrt(1 - self.beta_2 ** (1)) / (1 - self.beta_1 ** (1))
        else:
            self.lr_eff = (
                lr_val
                * np.sqrt(1 - self.beta_2 ** (self.iteration))
                / (1 - self.beta_1 ** (self.iteration))
            )

    def _log(self, fval, gradient, dx):
        """Function for creating a log entry of the optimization."""
        if self.log_file is not None:
            f = open(self.log_file, "a")
            if fval is not None:
                output = " %9d  %12.5f  %12.5f  %12.5f  %12.5f  %12.5f \n" % (
                    self.iteration,
                    fval,
                    np.linalg.norm(gradient),
                    dx,
                    self.lr_eff,
                    self.lr,
                )
            else:
                output = " %9d  %12.5f  %12.5f  %12.5f  %12.5f \n" % (
                    self.iteration,
                    np.linalg.norm(gradient),
                    dx,
                    self.lr_eff,
                    self.lr,
                )
            f.write(output)
            f.close()
