"""Quantum Kernel Optimizer"""

import numpy as np
from functools import partial
from typing import Optional, Sequence
from .kernel_optimization_base import KernelOptimizerBase
from .kernel_loss_base import KernelLossBase
from ...optimizers.optimizer_base import OptimizerBase


class KernelOptimizer(KernelOptimizerBase):
    """
    Quantum kernel optimizer.
    This class can be used to optimize the variational parameters of a quantum kernel.


    Args:
        loss (KernelLossBase): The loss function to be minimized.
        optimizer (OptimizerBase): The optimizer to be used.
        initial_parameters (Optional[Sequence[float]]): Initial parameters for the optimizer.


    **Example**

    .. code-block::

        from squlearn import Executor
        from squlearn.encoding_circuit import HubregtsenEncodingCircuit
        from squlearn.kernel.matrix import FidelityKernel
        from squlearn.optimizers import Adam
        from squlearn.kernel.optimization import NLL
        enc_circ = HubregtsenEncodingCircuit(num_qubits=num_qubits, num_features=num_features, num_layers=2)
        q_kernel = FidelityKernel(encoding_circuit=enc_circ, executor=Executor("statevector_simulator"))
        adam = Adam(options={"maxiter": 20, "lr": 0.1})
        nll_loss = NLL(quantum_kernel=q_kernel, sigma=noise_std**2)
        optimizer = KernelOptimizer(loss=nll_loss, optimizer=adam,
            initial_parameters=np.random.rand(enc_circ.num_parameters))
        opt_result = optimizer.run_optimization(X=X_train, y=Y_train)
        optimal_parameters = opt_result.x
        q_kernel.assign_parameters(optimal_parameters)

    Methods:
    ----------
    """

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
        if self._initial_parameters is None:
            self._initial_parameters = self._quantum_kernel.parameters

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
        self._optimal_value = opt_result.fun
        self._optimal_point = opt_result.x
        self._opt_result = opt_result

        return self._opt_result


# BACKUP FOR DOCUMENTATION
# Attributes:
#     ----------
#         loss (KernelLossBase): The loss function to be minimized.
#         optimizer (OptimizerBase): The optimizer to be used.
#         initial_parameters (Optional[Sequence[float]]): Initial parameters for the optimizer.
#         optimal_value (float): The optimal value of the loss function.
#         optimal_point (Sequence[float]): The optimal point of the loss function.
#         optimal_parameters (Sequence[float]): The optimal parameters of the quantum kernel.
