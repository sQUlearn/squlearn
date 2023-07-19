# kernel util
import numpy as np
import scipy
from sklearn.gaussian_process.kernels import Kernel
from ..matrix.kernel_matrix_base import KernelMatrixBase


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


def regularize_kernel(gram_matrix):
    """
    Thresholding regularization method of a Gram matrix (full or training kernel matrix)
    according to this `paper <https://arxiv.org/pdf/2105.02276.pdf>`_ to recover positive
    semi-definiteness. This method only changes the negative eigenvalues of the matrix by
    setting them to zero. This is done via a full eigenvalue decomposition, adjustment of
    the negative eigenvalues and composition of the adjusted spectrum and the original
    eigenvectors:

    .. math::
        D = V^T A V
        D'_{ij} = \text{max}\lbrace D_{ij},0 \rbrace
        R-THR(A) = V D' V^T

    This approach is equivalent to finding the positive semi-definite matrix closest to A
    in any unitarily invariant norm.

    Args:
        gram_matrix (np.ndarray) :
            Gram matrix.
    """
    evals, evecs = scipy.linalg.eig(gram_matrix)
    reconstruction = evecs @ np.diag(evals.clip(min=0)) @ evecs.T
    return np.real(reconstruction)


# deprecated regularization technique
def tikhonov_regularization(gram_matrix):
    """
    Tikhonov regularization method to recover positive semi-definiteness of a Gram matrix.
    In this method the spectrum of the matrix is displaced by its smallest eigenvalue
    :math:`\sigma_{min}` if it is negative, by subtracting it from all eigenvalues or
    equivalently from the diagonal.

    Args:
        gram_matrix (np.ndarray) :
            Gram matrix.
    """
    evals = scipy.linalg.eigvals(gram_matrix)
    if np.min(np.real(evals)) < 0:
        gram_matrix -= np.min(np.real(evals)) * np.identity(gram_matrix.shape[0])
    return gram_matrix


def regularize_full_kernel(K_train, K_testtrain, K_test):
    """
    Built full Gram matrix from training-, test-train- and test-test kernel matrices
    and the regularize full Gram matrix using the `regularize_kernel()` method.

    Args:
        K_train (np.ndarray) :
            Training kernel matrix of shape (n_train, n_train)
        K_testtrain (np.ndarray) :
            Test-Train kernel matrix of shape (n_test, n_train)
        K_test (np.ndarray) :
            Test kernel matrix of shape (n_test, n_test)
    """
    gram_matrix_total = np.block([[K_train, K_testtrain.T], [K_testtrain, K_test]])
    reconstruction = regularize_kernel(gram_matrix_total)
    print(f"Reconstruction error {np.sum(reconstruction - gram_matrix_total)}")

    K_train = reconstruction[: K_train.shape[0], : K_train.shape[1]]
    K_testtrain = reconstruction[K_train.shape[0] :, : K_testtrain.shape[1]]
    K_test = reconstruction[-K_test.shape[0] :, -K_test.shape[1] :]

    return K_train, K_testtrain, K_test
