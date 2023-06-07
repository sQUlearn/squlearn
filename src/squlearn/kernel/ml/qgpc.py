# QGPC
import numpy as np
from ..matrix import KernelMatrixBase
from .kernel_util import kernel_wrapper
from sklearn.gaussian_process import GaussianProcessClassifier


class QGPC(GaussianProcessClassifier):
    """Quantum Gaussian process regression (QGPC).

    Args:
    ---------
    quantum_kernel: KernelMatrixBase class quantum kernel object
    (either a fidelity kernel or the PQK must be provided)
    """

    def __init__(self, quantum_kernel: KernelMatrixBase, **kwargs):
        self._quantum_kernel = quantum_kernel
        super().__init__(**kwargs)
        self.kernel = kernel_wrapper(quantum_kernel)

    @property
    def quantum_kernel(self) -> KernelMatrixBase:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: KernelMatrixBase):
        """Sets quantum kernel"""
        self._quantum_kernel = quantum_kernel
        self.kernel = kernel_wrapper(quantum_kernel)
