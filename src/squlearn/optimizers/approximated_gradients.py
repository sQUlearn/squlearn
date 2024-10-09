"""Classes for approximated gradients"""

import numpy as np


class ApproxGradientBase:
    """Base class for evaluating approximated gradients"""

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Function that calculates the approximated gradient for given input x

        Args:
            x (np.ndarray): Input location at which the gradient is calculated

        Returns:
            Approximated gradient with the same dimension as x (np.ndarray)

        """
        raise NotImplementedError()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.gradient(x)


class FiniteDiffGradient(ApproxGradientBase):
    """
    Class for evaluating the finite differences gradient.

    Possible implementations are:

    Forward: [f(x+eps)-f(x)]/eps
    Backwards: [f(x)-f(x-eps)]/eps
    Central (default): [f(x+eps)-f(x-eps)]/2*eps
    Five-point: [-f(x+2eps)+8f(x+eps)-8f(x-eps)+f(x-2eps)]/12eps

    Args:
        fun (callable): Callable function for the gradient evaluation
        eps (float): Step for finite differences
        formula (str): type of finite differences. Possible values for type are
            'forward', 'backwards', 'central', and 'five-point'
    """

    def __init__(self, fun: callable, eps: float = 0.01, formula: str = "central") -> None:
        self.fun = fun
        self.eps = eps
        self.formula = formula

        if formula not in ("central", "forward", "backwards", "five-point"):
            raise ValueError("Wrong value of formula: " + formula)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Function that calculates the approximated gradient for given input x

        Args:
            x (np.ndarray): Input location at which the gradient is calculated

        Returns:
            Approximated gradient with the same dimension as x (np.ndarray)

        """
        if len(x.shape) != 1:
            raise ValueError("Unsupported shape of x!")

        if self.formula == "forward":
            f0 = self.fun(x)
            g = np.zeros(len(x))
            for i in range(len(x)):
                dx = np.eye(1, len(x), k=i)[0] * self.eps
                g[i] = ((self.fun(x + dx) - f0)) / self.eps  # Extract scalar

        elif self.formula == "backwards":
            f0 = self.fun(x)
            g = np.zeros(len(x))
            for i in range(len(x)):
                dx = np.eye(1, len(x), k=i)[0] * self.eps
                g[i] = ((f0 - self.fun(x - dx))) / self.eps  # Extract scalar

        elif self.formula == "central":
            g = np.zeros(len(x))
            for i in range(len(x)):
                dx = np.eye(1, len(x), k=i)[0] * self.eps
                g[i] = ((self.fun(x + dx) - self.fun(x - dx))) / (2.0 * self.eps)  # Extract scalar

        elif self.formula == "five-point":
            g = np.zeros(len(x))
            for i in range(len(x)):
                dx = np.eye(1, len(x), k=i)[0] * self.eps
                g[i] = (
                    -1.0 * self.fun(x + 2.0 * dx)
                    + 8.0 * self.fun(x + 1.0 * dx)
                    - 8.0 * self.fun(x - 1.0 * dx)
                    + 1.0 * self.fun(x - 2.0 * dx)
                ) / (
                    12.0 * self.eps
                )  # Extract scalar
        else:
            raise ValueError("Wrong value of type: " + self.formula)

        return g


class StochasticPerturbationGradient(ApproxGradientBase):
    """
    Class for evaluating the stochastic perturbation gradient estimation.

    g_i = f(x+eps*r)-f(x-eps*r)/2*eps*r_i  with random vector r

    This is used in the SPSA optimization.

    Args:
        fun (callable): Callable function for the gradient evaluation
        eps (float): Step for difference
        seed (int): Seed for the random vector generation
    """

    def __init__(self, fun: callable, eps: float = 0.1, seed: int = 0) -> None:
        self.fun = fun
        self.eps = eps
        self.rng = np.random.default_rng(seed=seed)

    def set_eps(self, eps) -> None:
        """Setter for the eps value (is often dynamically adjusted)"""
        self.eps = eps

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Function that calculates the approximated gradient for given input x

        Args:
            x (np.ndarray): Input location at which the gradient is calculated

        Returns:
            Approximated gradient with the same dimension as x (np.ndarray)

        """
        if len(x.shape) != 1:
            raise ValueError("Unsupported shape of x!")

        pert = self.rng.random(len(x))

        f1 = self.fun(x + self.eps * pert)
        f2 = self.fun(x - self.eps * pert)

        return np.divide(f1 - f2, 2.0 * self.eps * pert)
