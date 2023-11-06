import numpy as np
from typing import Optional, Sequence
from .kernel_loss_base import KernelLossBase
from ...optimizers.optimizer_base import OptimizerBase


class KernelOptimizerBase:
    """
    Empty parent class for defining a kernel optimizer object.

    Args:
        loss (KernelLossBase) :
            Loss function to be used for the kernel optimization
        optimizer (OptimizerBase) :
            Optimizer from squlearn.optimizers used for finding the minimum of the respective
            loss function.
        initial_parameters (Optional[Sequence[float]]) :
            Initial guess for the encoding circuit's trainable parameters which are to be optimized
    """

    def __init__(
        self,
        loss: KernelLossBase = None,
        optimizer: OptimizerBase = None,
        initial_parameters: Optional[Sequence[float]] = None,
    ) -> None:
        self._loss = loss
        self._optimizer = optimizer
        self._initial_parameters = initial_parameters

    def run_optimization(self, X: np.ndarray, y: np.ndarray = None):
        """
        Empty function to start running the actual optimization.

        Args:
            X (np.ndarray) :
                Data set features
            y (np.ndarray) :
                Data set labels
        """
        raise NotImplementedError
