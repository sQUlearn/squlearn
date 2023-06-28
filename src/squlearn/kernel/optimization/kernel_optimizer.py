import numpy as np
from functools import partial
from typing import Optional, Sequence
from .kernel_optimization_base import KernelOptimizerBase
from .kernel_loss_base import KernelLossBase
from ...optimizers.optimizer_base import OptimizerBase


class KernelOptimizer(KernelOptimizerBase):
    def __init__(
        self,
        loss: KernelLossBase = None,
        optimizer: OptimizerBase = None,
        initial_parameters: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(loss, optimizer, initial_parameters)
        self._quantum_kernel = loss._quantum_kernel
        self._opt_result = None
        self._optimier_evals = None
        self._optimal_value = None
        self._optimal_point = None
        self._optimal_parameters = None

    def run_optimization(self, x: np.ndarray, y: np.ndarray = None):
        
        num_params = self._quantum_kernel.num_parameters
        if num_params == 0:
            raise ValueError(
                "Quantum kernel cannot be fit because there are no training parameters specified."
            )

        # Perform kernel optimization
        loss_function = partial(self._loss.compute, data=x, labels=y)
        opt_result = self._optimizer.minimize(fun=loss_function, x0=self._initial_parameters)
        self._optimal_value = opt_result.fun
        self._optimal_point = opt_result.x
        self._opt_result = opt_result

        return self._opt_result
