import numpy as np
from abc import ABC, abstractmethod

from ..lowlevel_kernel.kernel_matrix_base import KernelMatrixBase


class KernelLossBase(ABC):
    """Empty parent class for a kernel loss function."""

    def __init__(self) -> None:
        self._quantum_kernel = None

    def set_quantum_kernel(self, quantum_kernel: KernelMatrixBase) -> None:
        """Set the quantum kernel matrix to be used in the loss.

        Args:
            quantum_kernel (KernelMatrixBase): The quantum kernel matrix to be used in the loss.
        """
        self._quantum_kernel = quantum_kernel

    @abstractmethod
    def compute(
        self,
        quantum_kernel: KernelMatrixBase,
        parameter_values: np.array,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute the target alignment loss.

        Args:
            quantum_kernel (KernelMatrixBase): The quantum kernel matrix to be used in the loss.
            parameter_values (np.ndarray): The parameter values for the variational quantum
                                           kernel parameters.
            data (np.ndarray): The training data to be used for the kernel matrix.
            labels (np.ndarray): The training labels.

        Returns:
            float: The loss value.
        """
        raise NotImplementedError
