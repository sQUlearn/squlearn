"""Quantum Kernel Optimizer"""

import numpy as np
from functools import partial
from typing import Optional, Sequence

from .kernel_matrix_base import KernelMatrixBase
from ..loss.kernel_loss_base import KernelLossBase
from ...optimizers.optimizer_base import OptimizerBase


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
        self._is_fitted = False

        self._loss.set_quantum_kernel(self._quantum_kernel)

    @property
    def is_fitted(self) -> bool:
        """Returns whether the quantum kernel has been fitted."""
        return self._is_fitted

    def run_optimization(self, X: np.ndarray, y: np.ndarray = None):
        """Run the optimization and return the result.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The labels.

        Returns:
            OptimizeResult: The optimization result.
        """

        if self._is_fitted:
            return None

        if self._quantum_kernel.num_parameters == 0:
            return None

        # Perform kernel optimization
        loss_function = partial(self._loss.compute, data=X, labels=y)
        opt_result = self._optimizer.minimize(fun=loss_function, x0=self._initial_parameters)
        self._optimal_parameters = opt_result.x

        # Assign optimal parameters to the quantum kernel
        self._quantum_kernel.assign_parameters(self._optimal_parameters)

        self._is_fitted = True

        return opt_result

    def assign_parameters(self, parameters: np.ndarray):
        """Set the training parameters of the encoding circuit to numerical values

        Args:
            parameters (np.ndarray): Array containing numerical values to be assigned to
                                     the trainable parameters of the encoding circuit
        """
        self._is_fitted = False
        self._quantum_kernel.assign_parameters(parameters)
        self._initial_parameters = parameters

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

    def set_params(self, **params):
        """Sets value of the kernel optimizer hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. ``num_qubits=2``
        """

        # Create dictionary of valid parameters
        valid_params = self.get_params(deep=True).keys()
        for key in params.keys():
            # Check if parameter is valid
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

        if "quantum_kernel" in params:
            self._quantum_kernel = params["quantum_kernel"]
            self._is_fitted = False
        if "loss" in params:
            self._loss = params["loss"]
            self._is_fitted = False
        if "optimizer" in params:
            self._optimizer = params["optimizer"]
            self._is_fitted = False
        if "initial_parameters" in params:
            self._initial_parameters = params["initial_parameters"]
            self._is_fitted = False

        # Set parameters of the Quantum Kernel and its underlying objects
        quantum_kernel_params = self._quantum_kernel.get_params().keys() & params.keys()
        if quantum_kernel_params:
            self._quantum_kernel.set_params(**{key: params[key] for key in quantum_kernel_params})
            self._is_fitted = False

        return self

    def get_params(self, deep=True) -> dict:
        """Returns hyper-parameters and their values of the fidelity kernel.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = {}
        params["quantum_kernel"] = self._quantum_kernel
        params["loss"] = self._loss
        params["optimizer"] = self._optimizer
        params["initial_parameters"] = self._initial_parameters

        if deep:
            params.update(self._quantum_kernel.get_params(deep=deep))

        return params
