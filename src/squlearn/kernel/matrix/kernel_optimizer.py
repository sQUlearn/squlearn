"""Quantum Kernel Optimizer"""

import numpy as np
from functools import partial
from typing import Optional, Sequence

from squlearn.kernel.matrix.kernel_matrix_base import KernelMatrixBase
from squlearn.kernel.optimization.kernel_loss_base import KernelLossBase
from squlearn.optimizers.optimizer_base import OptimizerBase


class KernelOptimizer(KernelMatrixBase):
    """
    Quantum kernel optimizer.
    This class can be used to optimize the variational parameters of a quantum kernel.

    Args:
        loss (KernelLossBase): The loss function to be minimized.
        optimizer (OptimizerBase): The optimizer to be used.
        quantum_kernel (KernelMatrixBase): The quantum kernel to be optimized.
        initial_parameters (Optional[Sequence[float]]): Initial parameters for the optimizer.
    """

    def __init__(
        self,
        quantum_kernel: KernelMatrixBase,
        loss: KernelLossBase = None,
        optimizer: OptimizerBase = None,
        initial_parameters: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(
            encoding_circuit=quantum_kernel.encoding_circuit,
            executor=quantum_kernel._executor,
            initial_parameters=initial_parameters,
            parameter_seed=quantum_kernel._parameter_seed,
            regularization=quantum_kernel._regularization,
        )
        self._quantum_kernel = quantum_kernel
        self._loss = loss
        self._optimizer = optimizer
        self._initial_parameters = (
            initial_parameters
            if initial_parameters is not None
            else self._quantum_kernel.parameters
        )
        self._optimal_parameters = None
        self._is_trainable = True

    def run_optimization(self, X: np.ndarray, y: np.ndarray = None):
        """Run the optimization and return the result.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The labels.

        Returns:
            OptimizeResult: The optimization result.
        """
        num_params = self._quantum_kernel.num_parameters
        if num_params == 0:
            raise ValueError(
                "Quantum kernel cannot be fit because there are no training parameters specified."
            )

        # Perform kernel optimization
        loss_function = partial(self._loss.compute, data=X, labels=y)
        opt_result = self._optimizer.minimize(fun=loss_function, x0=self._initial_parameters)
        self._optimal_parameters = opt_result.x

        # Assign optimal parameters to the quantum kernel
        self._quantum_kernel.assign_parameters(self._optimal_parameters)

        return opt_result

    def evaluate(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Evaluate the kernel matrix using the current parameters.

        Args:
            x (np.ndarray): Vector of training or test data.
            y (np.ndarray, optional): Vector of training or test data.

        Returns:
            np.ndarray: The evaluated kernel matrix.
        """
        return self._quantum_kernel.evaluate(x, y)

    def get_optimal_parameters(self) -> np.ndarray:
        """Get the optimal parameters.

        Returns:
            np.ndarray: The optimal parameters.
        """
        return self._optimal_parameters
