"""Quantum Kernel Ridge Regressor"""
from ..matrix.kernel_matrix_base import KernelMatrixBase

import scipy
import numpy as np
from typing import Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin

from .kernel_util import regularize_kernel, tikhonov_regularization

# from ..kernel_util import KernelRegularizer, DepolarizingNoiseMitigation


class QKRR(BaseEstimator, RegressorMixin):
    """
    Quantum Kernel Ridge Regression

    This class implements the kernel ridge regression analogous to sklearn
    but is not a wrapper.

    Args:
        quantum_kernel (KernelMatrixBase): The quantum kernel matrix to be used in the KRR pipeline
            (either a fidelity quantum kernel (FQK) or projected quantum kernel (PQK) must be provided)
        alpha (Union[float, np.ndarray]), default=1.e-6: Hyperparameter for the regularization strength; must be a positive float.
            This regularization improves the conditioning of the problem and assure the solvability of the resulting
            linear system. Larger values specify stronger regularization (cf., e.g., `https://en.wikipedia.org/wiki/Ridge_regression´)
        regularize (Union[str, None]), default=None: Option for choosing different regularization techniques ('thresholding' or 'tikhonov')
            after `http://arxiv.org/pdf/2105.02276v1´ for the training kernel matrix, prior to solving the linear system in the fit() procedure
    """

    def __init__(
        self,
        quantum_kernel: Optional[KernelMatrixBase] = None,
        alpha: Union[float, np.ndarray] = 1.0e-6,
        regularize: Union[str, None] = None,
    ) -> None:
        self._quantum_kernel = quantum_kernel  # May be worth to set FQK as default here?
        self.alpha = alpha
        self._regularize = regularize
        self.x_train = None
        self.k_testtrain = None
        self.k_train = None
        self.dual_coeff_ = None
        self.num_qubits = self._quantum_kernel.num_qubits

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.k_train = self._quantum_kernel.evaluate(x=self.x_train)  # set up kernel matrix
        # check if regularize argument is set and define corresponding method
        if self._regularize is not None:
            if self._regularize == "thresholding":
                self.k_train = regularize_kernel(self.k_train)
            elif self._regularize == "tikhonov":
                self.k_train = tikhonov_regularization(self.k_train)

        self.k_train = self.k_train + self.alpha * np.eye(self.k_train.shape[0])

        # solution of KRR problem obtained by solving linear system K*\alpha = y
        # using Cholesky decomposition for providing numerical stability
        try:
            L = scipy.linalg.cholesky(self.k_train, lower=True)
            self.dual_coeff_ = scipy.linalg.cho_solve((L, True), y_train)
        except np.linalg.LinAlgError:
            print("Increase regularization parameter alpha")

        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self.k_train is None:
            raise ValueError("The fit() method has to be called beforehand.")

        self.k_testtrain = self._quantum_kernel.evaluate(x_test, self.x_train)
        prediction = np.dot(self.k_testtrain, self.dual_coeff_)
        return prediction

    # All scikit-learn estimators have get_params and set_params
    # (cf. https://scikit-learn.org/stable/developers/develop.html)
    def get_params(self, deep: bool = True):
        return {
            "quantum_kernel": self._quantum_kernel,
            "alpha": self.alpha,
            "regularize": self._regularize,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
