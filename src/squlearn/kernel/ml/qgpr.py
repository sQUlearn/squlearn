from ..matrix.kernel_matrix_base import KernelMatrixBase
from .helper_functions import stack_input
from .kernel_util import regularize_full_kernel, tikhonov_regularization

import numpy as np
from scipy.linalg import cholesky, cho_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing._data import _handle_zeros_in_scale

##### change location pointer of helper functions ########
from numbers import Real


class QGPR(BaseEstimator, RegressorMixin):
    """Quantum Gaussian process regression (QGPR).

    Args:
    ---------
    quantum_kernel: KernelMatrixBase class quantum kernel object
    (either a fidelity kernel or the PQK must be provided)
    sigma: float; regularization parameter that gets added to the training kernels main diagonal
    normalize_y: bool; enable normalization of y-variable. Default = False.
    regularize: string; enable full gram matrix regularization technique via 'full'. Default = 'off'.
    or enable Tikhonov regularization via 'tikhonov'.

    """

    def __init__(
        self,
        quantum_kernel: KernelMatrixBase,
        sigma=0.0,
        normalize_y=False,
        regularize="full",
    ):
        self._quantum_kernel = quantum_kernel
        self.X_train = None
        self.K_train = None
        self.K_test = None
        self.K_testtrain = None
        self._L = None
        self._alpha = None
        self.y_train = None
        self.sigma = sigma
        self.normalize_y = normalize_y
        self.regularize = regularize

    def fit(self, X_train, y_train):
        # stack inpur as many times as nr_qubits
        if self._quantum_kernel.num_features > 1:
            self.X_train = stack_input(
                x_vec=X_train, num_features=self._quantum_kernel.num_features
            )
        else:
            self.X_train = X_train
        self.K_train = self._quantum_kernel.evaluate(x=self.X_train)
        if self.normalize_y:
            self._y_train_mean = np.mean(y_train, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y_train, axis=0), copy=False)
            self.y_train = (y_train - self._y_train_mean) / self._y_train_std
        else:
            self.y_train = y_train
        return self

    # needed to be an "official" sklearn estimator
    def get_params(self, deep=True):
        return {
            "quantum_kernel": self._quantum_kernel,
            "sigma": self.sigma,
            "normalize_y": self.normalize_y,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X_test, return_std=True, return_cov=False):
        if self.K_train is None:
            raise ValueError("There is no training data. Please call the fit method first.")
        if self._quantum_kernel.num_features > 1:
            X_test = stack_input(x_vec=X_test, num_features=self._quantum_kernel.num_features)
        self.K_test = self._quantum_kernel.evaluate(x=X_test)

        self.K_testtrain = self._quantum_kernel.evaluate(x=X_test, y=self.X_train)

        if self.regularize == "full":
            print("Regularizing full Gram matrix")
            self.K_train, self.K_testtrain, self.K_test = regularize_full_kernel(
                self.K_train, self.K_testtrain, self.K_test
            )
        elif self.regularize == "tikhonov":
            print("Regularizing Gram matrix with Tikhonov")
            self.K_train = tikhonov_regularization(self.K_train)
            self.K_test = tikhonov_regularization(self.K_test)
        self.K_train += self.sigma * np.identity(self.K_train.shape[0])
        try:
            self._L = cholesky(self.K_train, lower=True)
        except np.linalg.LinAlgError:
            print("corrected the train matrix a bit")
            self.K_train += 1e-8 * np.identity(self.K_train.shape[0])
            self._L = cholesky(self.K_train, lower=True)
        self._alpha = cho_solve((self._L, True), self.y_train)
        mean, cov = self.calculate_cov_and_mean()

        # undo normalization
        if self.normalize_y:
            mean = self._y_train_std * mean + self._y_train_mean
        if return_std:
            std = np.sqrt(np.diag(cov))
            return mean, std
        if return_cov:
            return mean, cov
        else:
            return mean

    def calculate_cov_and_mean(self):
        QGP_mean = self.K_testtrain.dot(self._alpha)
        v = cho_solve((self._L, True), self.K_testtrain.T)
        QGP_cov = self.K_test - (self.K_testtrain @ v)
        return QGP_mean, QGP_cov
