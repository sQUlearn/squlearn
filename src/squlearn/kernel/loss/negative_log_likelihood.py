"""Negative log likelihood loss function"""

import scipy
import numpy as np

from typing import Sequence
from .kernel_loss_base import KernelLossBase
from ..lowlevel_kernel.kernel_matrix_base import KernelMatrixBase


class NLL(KernelLossBase):
    r"""
    Negative log likelihood loss function.
    This class can be used to compute the negative log likelihood loss function
    for a given quantum kernel
    :math:`K_{\theta}` with variational parameters :math:`\theta`.
    The definition of the function is taken from Equation 5.8 Chapter 5.4 of Ref. [1].

    The log-likelihood function is defined as:

    .. math::

        L(\theta) = -\frac{1}{2} log(|K_{\theta} + \sigmaI|)-\frac{1}{2} y^{T}(K_{\theta}
        + \sigmaI)^{-1}y-\frac{n}{2} log(2\pi)

    Args:
        sigma: (float), default=0.0: Hyperparameter for the regularization strength.

    References
    ----------
        [1]: `Carl E. Rasmussen and Christopher K.I. Williams,
        "Gaussian Processes for Machine Learning",
        MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

    Methods:
    --------
    """

    def __init__(self, sigma=0.0):
        super().__init__()
        self._sigma = sigma

    def compute(
        self,
        parameter_values: np.ndarray,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute the negative log likelihood loss function.

        Args:
            parameter_values (np.ndarray): The parameter values for the variational quantum
                                           kernel parameters.
            data (np.ndarray): The training data to be used for the kernel matrix.
            labels (np.ndarray): The training labels.

        Returns:
            float: The negative log likelihood loss value.
        """

        if self._quantum_kernel is None:
            raise ValueError(
                "Quantum kernel is not set, please set the quantum kernel with set_quantum_kernel method"
            )

        # Bind training parameters
        self._quantum_kernel.assign_parameters(parameter_values)

        # get estimated kernel matrix
        kmatrix = self._quantum_kernel.evaluate(data)
        kmatrix = kmatrix + self._sigma * np.eye(kmatrix.shape[0])

        # Cholesky decomposition since numerically more stable
        L = scipy.linalg.cholesky(kmatrix, lower=True)
        S1 = scipy.linalg.solve_triangular(L, labels, lower=True)
        S2 = scipy.linalg.solve_triangular(L.T, S1, lower=False)
        neg_log_lh = (
            np.sum(np.log(np.diagonal(L)))
            + 0.5 * labels.T @ S2
            + 0.5 * len(data) * np.log(2.0 * np.pi)
        )
        neg_log_lh = neg_log_lh.reshape(-1)

        return neg_log_lh
