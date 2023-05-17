"""Optimization methods for QNNs
"""

# Authors: Moritz Willmann <moritz.willmann@ipa.fraunhofer.de>
# License: ...

import abc
import numpy as np


class OptimizerResult:  # TODO: maybe scipy class?
    """Class for holding the final result of the optimization"""

    def __init__(self):
        self.x = None
        self.nit = 0
        self.fun = 0.0


class OptimizerBase(abc.ABC):
    """Base class for QNN optimizers."""

    def minimize(
        self, fun, x0, grad=None, bounds=None  # pylint: disable=invalid-name
    ) -> OptimizerResult:
        """Minimize a function"""
        raise NotImplementedError()


class IterativeOptimizerMixin:
    """Mixin for iterative optimizers."""

    def __init__(self) -> None:
        self.iteration = 0

    def step(self, **kwargs):
        """Perform one update step."""
        raise NotImplementedError()


class SGDMixin(IterativeOptimizerMixin, abc.ABC):
    """Mixin for stochastic gradient descent based optimizers."""

    def step(self, **kwargs):
        """ "

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

    @abc.abstractmethod
    def _get_update(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def _update_lr(self) -> None:
        raise NotImplementedError()
