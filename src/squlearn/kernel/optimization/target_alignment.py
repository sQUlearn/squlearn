"""Target alignment loss function for kernel matrices."""

import numpy as np

from typing import Sequence
from .kernel_loss_base import KernelLossBase
from ..matrix.kernel_matrix_base import KernelMatrixBase


class TargetAlignment(KernelLossBase):
    """
    Target alignment loss function.
    This class can be used to compute the target alignment for a given quantum kernel
    :math:`K_{θ}` with variational parameters :math:`θ`.
    The definition of the function is taken from Equation (27,28) of [1].
    The log-likelihood function is defined as:

    .. math::

        TA(K_{θ}) =
        \\frac{\\sum_{i,j} K_{θ}(x_i, x_j) y_i y_j}
        {\\sqrt{\\sum_{i,j} K_{θ}(x_i, x_j)^2 \\sum_{i,j} y_i^2 y_j^2}}

    Args:
        quantum_kernel (KernelMatrixBase): The quantum kernel to be used
            (either a fidelity quantum kernel (FQK)
            or projected quantum kernel (PQK) must be provided).

    References
    -----------
        [1]: T. Hubregtsen et al.,
        "Training Quantum Embedding Kernels on Near-Term Quantum Computers",
        `arXiv:2105.02276v1 (2021) <https://arxiv.org/abs/2105.02276>`_.

    Methods:
    --------
    """

    def __init__(self, quantum_kernel: KernelMatrixBase):
        super().__init__(quantum_kernel)

    def compute(
        self,
        parameter_values: Sequence[float],
        data: np.ndarray,
        labels: np.ndarray,
        rescale_class_labels=True,
    ) -> float:
        """Compute the target alignment.

        Args:
            parameter_values: (Sequence[float]):
                The parameter values for the variational quantum kernel parameters.
            data (np.ndarray): The  training data to be used for the kernel matrix.
            labels (np.ndarray): The training labels.
            rescale_class_labels: (bool), default=True:
                Whether to rescale the class labels to -1 and 1.

        Returns:
            float: The negative target alignment.
        """

        # Bind training parameters
        self._quantum_kernel.assign_parameters(parameter_values)

        # Get estimated kernel matrix
        kmatrix = self._quantum_kernel.evaluate(data)
        if rescale_class_labels:
            nplus = np.count_nonzero(np.array(labels) == 1)
            nminus = len(labels) - nplus
            _Y = np.array([y / nplus if y == 1 else y / nminus for y in labels])
        else:
            _Y = np.array(labels)

        T = np.outer(_Y, _Y)
        inner_product = np.sum(kmatrix * T)
        norm = np.sqrt(np.sum(kmatrix * kmatrix) * np.sum(T * T))
        alignment = inner_product / norm
        return -alignment
