"""Quantum Gaussian Process Regressor"""
import warnings

from ..matrix.kernel_matrix_base import KernelMatrixBase
from ..matrix.regularization import regularize_full_kernel

import numpy as np
from scipy.linalg import cholesky, cho_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing._data import _handle_zeros_in_scale


class QGPR(BaseEstimator, RegressorMixin):
    """
    Quantum Gaussian Process Regression (QGPR).

    This class implements the Gaussian process regression analogous to scikit-learn
    but is not a wrapper.
    The implementation is based on Algorithm 2.1 of Ref. [1].

    Args:
        quantum_kernel (KernelMatrixBase):
                The quantum kernel matrix to be used for the Gaussian process
                (either a fidelity quantum kernel (FQK) or
                projected quantum kernel (PQK) must be provided)
        sigma: (float), default=1.e-6: Hyperparameter for the regularization strength;
                must be a positive float.
                This regularization improves the conditioning of the problem
                and assure the solvability of the resulting
                linear system. Larger values specify stronger regularization.
        normalize_y: (bool), default=False: Whether to normalize
                the target values y by removing the mean and scaling to
                unit-variance. This is recommended for cases where zero-mean,
                unit-variance priors are used. Note that, in this implementation,
                the normalization is reversed before the GP predictions are reported.
        full_regularization: (bool), default=True: enable full gram matrix regularization.

    See Also
    --------
        squlearn.kernel.ml.QKRR : Quantum Gaussian Process regression.
        squlearn.kernel.ml.QSVR : Quantum Support Vector regression.

    References
    ----------
       [1]: `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

       [2]: F.Rapp, M.Roth "Quantum Gaussian Process Regression for Bayesian Optimization",
       `<https://arxiv.org/pdf/2304.12923.pdf>`_.


    **Example**

    .. code-block::

        from squlearn import Executor
        from squlearn.feature_map import QEKFeatureMap
        from squlearn.kernel.matrix import FidelityKernel
        from squlearn.kernel.ml import QGPR
        fmap = QEKFeatureMap(num_qubits=num_qubits, num_features=num_features, num_layers=2)
        q_kernel = FidelityKernel(feature_map=fmap, executor=Executor("statevector_simulator"))
        q_kernel.assign_parameters(np.random.rand(fmap.num_parameters))
        qgpr_ansatz = QGPR(quantum_kernel=q_kernel)
        qgpr_ansatz.fit(sample_train,label_train)
        qgpr_ansatz.predict(sample_test)

    Methods:
    --------
    """

    def __init__(
        self,
        quantum_kernel: KernelMatrixBase,
        sigma: float = 1.0e-6,
        normalize_y: bool = False,
        full_regularization: bool = True,
        **kwargs,
    ):
        self._quantum_kernel = quantum_kernel
        self.X_train = None
        self.y_train = None
        self.sigma = sigma
        self.normalize_y = normalize_y
        self._full_regularization = full_regularization

        self.K_train = None
        self.K_test = None
        self.K_testtrain = None
        self._L = None
        self._alpha = None

        # Apply kwargs to set_params
        update_params = self.get_params().keys() & kwargs.keys()
        if update_params:
            self.set_params(**{key: kwargs[key] for key in update_params})

    def fit(self, X_train, y_train):
        """Fit Quantum Gaussian process regression model.
        The fit method of the QGPR class just calculates the training kernel matrix.
        Depending on the choice of normalize_y the target values are normalized.

        Args:
            X_train: The training data of shape (n_samples, n_features)
            y_train: Target values in training data of shape (n_samples,)

        Returns:
            self: object
            QuantumGaussianProcessRegressor class instance.
        """
        self.X_train = X_train
        if self._full_regularization:
            if self._quantum_kernel._regularization is not None:
                warnings.warn(
                    f"The regularization of the quantum kernel is set to"
                    f" {self._quantum_kernel._regularization}. If full_regularization"
                    f"is True, best results are achieved with no additional quantum "
                    f"kernel regularization."
                )
            self.K_train = self._quantum_kernel.evaluate(x=self.X_train)
        else:
            self.K_train = self._quantum_kernel.evaluate(x=self.X_train)

        if self.normalize_y:
            self._y_train_mean = np.mean(y_train, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y_train, axis=0), copy=False)
            self.y_train = (y_train - self._y_train_mean) / self._y_train_std
        else:
            self.y_train = y_train
        return self

    def predict(self, X_test, return_std=False, return_cov=False):
        """Predict using the  Quantum Gaussian process regression model.
        Depending on the choice of regularization the quantum kernel matrix is regularized.
        The respective solution of the QKRR problem
        is obtained by solving the linear system using scipy's Cholesky decomposition for
        providing numerical stability
        Optionally also
        returns its standard deviation (`return_std=True`) or covariance
        (`return_cov=True`). Note that at most one of the two can be requested.

        Args:
            X_test: The test data of shape (n_samples, n_features)
            return_std: (bool),
                default=True: Whether to return the standard deviation of the prediction
            return_cov: (bool), default=False:
                Whether to return the covariance of the prediction

        Returns:
            y_mean: The predicted values of shape (n_samples,)
                Mean of predictive distribution at query points.
            y_std: The standard deviation of the prediction of shape (n_samples,), optional
                Standard deviation of predictive distribution at query points.
                Only returned when `return_std` is True.
            y_cov: The covariance of the prediction of shape (n_samples, n_samples), optional
                Covariance of joint predictive distribution a query points.
                Only returned when `return_cov` is True.
        """

        if self.K_train is None:
            raise ValueError("There is no training data. Please call the fit method first.")

        self.K_test = self._quantum_kernel.evaluate(x=X_test)
        self.K_testtrain = self._quantum_kernel.evaluate(x=X_test, y=self.X_train)
        if self._full_regularization:
            print("Regularizing full Gram matrix")
            self.K_train, self.K_testtrain, self.K_test = regularize_full_kernel(
                self.K_train, self.K_testtrain, self.K_test
            )

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
        """Calculates the mean and covariance of the QGPR model"""
        QGP_mean = self.K_testtrain.dot(self._alpha)
        v = cho_solve((self._L, True), self.K_testtrain.T)
        QGP_cov = self.K_test - (self.K_testtrain @ v)
        return QGP_mean, QGP_cov

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the QGPR method.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = {
            "quantum_kernel": self._quantum_kernel,
            "sigma": self.sigma,
            "normalize_y": self.normalize_y,
            "full_regularization": self._full_regularization,
        }
        if deep:
            params.update(self._quantum_kernel.get_params(deep=deep))
        return params

    def set_params(self, **params) -> None:
        """
        Sets value of the feature map hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. num_qubits=2.
        """
        valid_params = self.get_params()
        valid_params_qgpr = self.get_params(deep=False)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

            # Set parameters of the QGPR
            if key in valid_params_qgpr:
                try:
                    setattr(self, key, value)
                except:
                    setattr(self, "_" + key, value)

        # Set parameters of the Quantum Kernel and its underlying objects
        param_dict = {}
        valid_params = self._quantum_kernel.get_params().keys()
        for key, value in params.items():
            if key in valid_params:
                param_dict[key] = value
        if len(param_dict) > 0:
            self._quantum_kernel.set_params(**param_dict)
        return self
