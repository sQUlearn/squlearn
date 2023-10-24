"""Optimization methods in sQUlearn."""

import abc
import numpy as np


def default_callback(*args):
    """Default callback function."""
    pass


class OptimizerResult:  # TODO: maybe scipy class?
    """Class for holding the final result of the optimization"""

    def __init__(self):
        self.x = None
        self.nit = 0
        self.fun = 0.0


class OptimizerBase(abc.ABC):
    """Base class for QNN optimizers."""

    def minimize(
        self,
        fun: callable,
        x0: np.ndarray,
        grad: callable = None,
        bounds=None,  # pylint: disable=invalid-name
    ) -> OptimizerResult:
        """Function to minimize a given function.

        Args:
            fun (callable): Function to minimize.
            x0 (numpy.ndarray): Initial guess.
            grad (callable): Gradient of the function to minimize.
            bounds (sequence): Bounds for the parameters.

        Returns:
            Result of the optimization in class:`OptimizerResult` format.
        """
        raise NotImplementedError()

    def set_callback(self, callback):
        """Set the callback function."""
        self.callback = callback


class IterativeMixin:
    """Mixin for iteration based optimizers."""

    def __init__(self):
        self.iteration = 0


class StepwiseMixin(IterativeMixin):
    """Mixin for optimizer for which we can execute single steps."""

    def step(self, **kwargs):
        """Perform one update step."""
        raise NotImplementedError()


class SGDMixin(StepwiseMixin, abc.ABC):
    """Mixin for stochastic gradient descent based optimizers."""

    def step(self, **kwargs):
        """Perform one update step.

        Args:
            x: Current value
            grad: Precomputed gradient

        Returns:
            Updated x
        """
        if "x" in kwargs:
            x = kwargs["x"]
        else:
            raise TypeError("x argument is missing in step function.")
        if "grad" in kwargs:
            grad = kwargs["grad"]
        else:
            raise TypeError("grad argument is missing in step function.")

        update = self._get_update(grad)
        x_return = x + update
        self.iteration += 1
        self._update_lr()
        return x_return

    def reset(self):
        """
        Resets the object to its initial state.

        This function does not take any parameters.

        Returns:
            None: This function does not return anything.
        """
        pass

    @abc.abstractmethod
    def _get_update(self, grad: np.ndarray) -> np.ndarray:
        """Function that returns the update for a given gradient."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _update_lr(self) -> None:
        """Function for updating the learning rate."""
        raise NotImplementedError()
