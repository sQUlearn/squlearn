"""Target alignment loss function for kernel matrices."""

import numpy as np

from .kernel_loss_base import KernelLossBase
from ..lowlevel_kernel.kernel_matrix_base import KernelMatrixBase


class TargetAlignment(KernelLossBase):
    r"""
    Target alignment loss function.
    This class can be used to compute the target alignment for a given quantum kernel
    :math:`K_{\theta}` with variational parameters :math:`\theta`.
    The definition of the function is taken from Equation (27,28) of [1].
    The target alignment loss is defined as:

    .. math::

        TA(K_{\theta}) =
        \frac{\\um_{i,j} K_{\theta}(x_i, x_j) y_i y_j}
        {\sqrt{\sum_{i,j} K_{\theta}(x_i, x_j)^2 \sum_{i,j} y_i^2 y_j^2}}

    Args:
        rescale_class_labels (bool): Whether to rescale the class labels to -1 and 1
                                     (default=True).

    References
    -----------
        [1]: T. Hubregtsen et al.,
        "Training Quantum Embedding Kernels on Near-Term Quantum Computers",
        `arXiv:2105.02276v1 (2021) <https://arxiv.org/abs/2105.02276>`_.

    Methods:
    --------
    """

    def __init__(self, rescale_class_labels=True) -> None:
        """ """
        super().__init__()
        self._rescale_class_labels = rescale_class_labels

    def compute(
        self,
        parameter_values: np.ndarray,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute the target alignment loss.

        Args:
            parameter_values (np.ndarray): The parameter values for the variational quantum kernel
                                           parameters.
            data (np.ndarray): The  training data to be used for the kernel matrix.
            labels (np.ndarray): The labels of the training data.

        Returns:
            float: The negative target alignment.
        """

        if self._quantum_kernel is None:
            raise ValueError(
                "Quantum kernel is not set, please set the quantum kernel with set_quantum_kernel method"
            )

        # Bind training parameters
        self._quantum_kernel.assign_parameters(parameter_values)

        # Get estimated kernel matrix
        kmatrix = self._quantum_kernel.evaluate(data)
        if self._rescale_class_labels:
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
