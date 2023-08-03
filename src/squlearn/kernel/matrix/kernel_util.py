# kernel util
import numpy as np

from sklearn.gaussian_process.kernels import Kernel

from .kernel_matrix_base import KernelMatrixBase


def kernel_wrapper(kernel_matrix: KernelMatrixBase):
    """
    Wrapper for sQUlearn's KernelMatrixBase to scikit-learn kernel objects.

    Args:
        kernel_matrix (KernelMatrixBase) :
            Quantum kernel matrix which is to be wrapped into scikit-learn kernel
    """

    class CustomKernel(Kernel):
        def __init__(self, kernel_matrix: KernelMatrixBase):
            self.kernel_matrix = kernel_matrix
            super().__init__()

        def __call__(self, X, Y=None, eval_gradient=False):
            if Y is None:
                Y = X
            kernel_matrix = self.kernel_matrix.evaluate(X, Y)
            if eval_gradient:
                raise NotImplementedError("Gradient not yet implemented for this kernel.")
            else:
                return kernel_matrix

        def diag(self, X):
            return np.diag(self.kernel_matrix.evaluate(X))

        @property
        def requires_vector_input(self):
            return True

        def is_stationary(self):
            return self.kernel_matrix.is_stationary()

    return CustomKernel(kernel_matrix)
