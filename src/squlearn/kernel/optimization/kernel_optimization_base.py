import numpy as np
from typing import Optional, Sequence
from .kernel_loss_base import KernelLossBase
from ...optimizers.optimizer_base import OptimizerBase


class KernelOptimizerBase:
    def __init__(
        self,
        loss: KernelLossBase = None,
        optimizer: OptimizerBase = None,
        initial_parameters: Optional[Sequence[float]] = None,
    ) -> None:
        self._loss = loss
        self._optimizer = optimizer
        self._initial_parameters = initial_parameters

    def run_optimization(self, x: np.ndarray, y: np.ndarray = None):
        raise NotImplementedError
