import abc
from collections import deque
import numpy as np

from .optimizer_base import OptimizerBase, SGDMixin, OptimizerResult
from .approximated_gradients import FiniteDiffGradient


class Adam(OptimizerBase, SGDMixin):
    """sQUlearn's implementation of the ADAM optimizer"""

    def __init__(self, options: dict = None) -> None:  # TODO: kwargs?
        super(SGDMixin, self).__init__()  #  set-up of self.iterations=0

        if options is None:
            options = {}

        self.tol = options.get("tol", 1e-6)
        self.lr = options.get("lr", 0.05)
        self.beta_1 = options.get("beta_1", 0.9)
        self.beta_2 = options.get("beta_2", 0.99)
        self.noise_factor = options.get("noise_factor", 1e-8)
        self.num_average = options.get("num_average", 1)
        self.maxiter = options.get("maxiter", 100)
        self.maxiter_total = options.get("maxiter_total", self.maxiter)
        self.log_file = options.get("log_file", None)
        self.skip_fun = options.get("skip_fun", False)
        self.eps = options.get("eps", 0.01)

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
        self.gradient_deque = deque(maxlen=self.num_average)
        self.m = None
        self.v = None
        self.x = None
        self.lr_eff = 0.0
        self.iteration = 0
        self._update_lr()

    def minimize(self, fun, x0, grad=None, bounds=None) -> OptimizerResult:
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
        if self.m is None:
            self.m = np.zeros(np.shape(grad))

        if self.v is None:
            self.v = np.zeros(np.shape(grad))

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * grad * grad

        return (
            -1.0 * self.lr_eff * self.m.flatten() / (np.sqrt(self.v.flatten()) + self.noise_factor)
        )

    def _update_lr(self):
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
