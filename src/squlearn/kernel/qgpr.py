"""Quantum Gaussian Process Regression"""

import warnings
from packaging import version

import numpy as np
from typing import Optional, Union
from scipy.linalg import lu_factor, lu_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing._data import _handle_zeros_in_scale

from sklearn import __version__

if version.parse(__version__) >= version.parse("1.6"):
    from sklearn.utils.validation import validate_data
else:

    def validate_data(self, *args, **kwargs):
        return self._validate_data(*args, **kwargs)


from .lowlevel_kernel.kernel_matrix_base import KernelMatrixBase
from .lowlevel_kernel.regularization import regularize_full_kernel


class QGPR(BaseEstimator, RegressorMixin):
    """
    Quantum Gaussian Process Regression (QGPR).

    This class implements the Gaussian process regression analogous to scikit-learn
    but is not a wrapper.
    The implementation is based on Algorithm 2.1 of Ref. [1].
    Additional arguments can be set via ``**kwargs``.

    Args:
        quantum_kernel (Optional[Union[KernelMatrixBase, str]]):
            The quantum kernel matrix to be used for the Gaussian process
            (either a fidelity quantum kernel (FQK) or
            projected quantum kernel (PQK) must be provided). By
            setting quantum_kernel="precomputed", X is assumed to be a kernel matrix
            - train for fit() method and total Gram matrix within predict(). This
            is particularly useful when storing quantum kernel
            matrices from real backends to numpy arrays.
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
        **kwargs: Keyword arguments for the quantum kernel matrix, possible arguments can be obtained
            by calling ``get_params()``. Can be used to set for example the number of qubits
            (``num_qubits=``), or (if supported) the number of layers (``num_layers=``)
            of the underlying encoding circuit.

    See Also
    --------
        squlearn.kernel.QKRR : Quantum Gaussian Process regression.
        squlearn.kernel.QSVR : Quantum Support Vector regression.

    References
    ----------
       [1]: `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

       [2]: F.Rapp, M.Roth "Quantum Gaussian Process Regression for Bayesian Optimization",
       `<https://link.springer.com/article/10.1007/s42484-023-00138-9>`_.


    **Example**

    .. code-block::

        from squlearn import Executor
        from squlearn.encoding_circuit import HubregtsenEncodingCircuit
        from squlearn.kernel.lowlevel_kernel import FidelityKernel
        from squlearn.kernel import QGPR
        enc_circ = HubregtsenEncodingCircuit(num_qubits=num_qubits, num_features=num_features, num_layers=2)
        q_kernel = FidelityKernel(encoding_circuit=enc_circ, executor=Executor())
        q_kernel.assign_parameters(np.random.rand(enc_circ.num_parameters))
        qgpr_ansatz = QGPR(quantum_kernel=q_kernel)
        qgpr_ansatz.fit(sample_train,label_train)
        qgpr_ansatz.predict(sample_test)

    Methods:
    --------
    """

    def __init__(
        self,
        quantum_kernel: Optional[Union[KernelMatrixBase, str]] = None,
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
        self.full_regularization = full_regularization

        self.K_train = None
        self.K_test = None
        self.K_testtrain = None
        self._LU = None
        self._piv = None
        self._alpha = None

        # Apply kwargs to set_params
        update_params = self.get_params().keys() & kwargs.keys()
        if update_params:
            self.set_params(**{key: kwargs[key] for key in update_params})

    def fit(self, X, y):
        """Fit Quantum Gaussian process regression model.
        The fit method of the QGPR class just calculates the training kernel matrix.
        Depending on the choice of normalize_y the target values are normalized.

        Args:
            X: array-like or sparse matrix of shape (n_samples, n_features)
                The training data.
                If quantum_kernel == "precomputed" this is instead a precomputed training kernel
                matrix of shape (n_samples, n_samples)
            y: array-like of shape (n_samples,)
                Target values.

        Return:
            Returns an instance of self.
        """

        X, y = validate_data(
            self,
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=True,
            dtype="numeric",
        )

        self.X_train = X

        if isinstance(self._quantum_kernel, str):
            if self._quantum_kernel == "precomputed":
                self.K_train = X
            else:
                raise ValueError("Unknown quantum kernel: {}".format(self._quantum_kernel))
        elif isinstance(self._quantum_kernel, KernelMatrixBase):

            # check if quantum kernel is trainable
            if self._quantum_kernel.is_trainable:
                self._quantum_kernel.run_optimization(self.X_train, y)

            if self.full_regularization:
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
        else:
            raise ValueError(
                "Unknown type of quantum kernel: {}".format(type(self._quantum_kernel))
            )

        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)
            self.y_train = (y - self._y_train_mean) / self._y_train_std
        else:
            self.y_train = y
        return self

    def predict(self, X: np.ndarray, return_std=False, return_cov=False):
        """
        Predict using the  Quantum Gaussian process regression model.
        Depending on the choice of regularization the quantum kernel matrix is regularized.
        The respective solution of the QKRR problem
        is obtained by solving the linear system using scipy's LU decomposition for
        providing numerical stability
        Optionally also
        returns its standard deviation (`return_std=True`) or covariance
        (`return_cov=True`). Note that at most one of the two can be requested.

        Args:
            X: The test data of shape (n_samples, n_features). If
                quantum_kernel == "precomputed", this is the precomputed
                Gram matrix instead, which has to be of shape
                np.block[[K_train, K_testtrain.T], [K_testtrain, K_test]]
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

        X = validate_data(self, X, ensure_2d=True, dtype="numeric", reset=False)

        if self.K_train is None:
            raise ValueError("There is no training data. Please call the fit method first.")
        if return_std and return_cov:
            raise ValueError(
                "Only one of return_std or return_cov can be True. " "Currently both are True."
            )
        if isinstance(self._quantum_kernel, str):
            if self._quantum_kernel == "precomputed":
                warnings.warn(
                    "Since the `precomputed´ option is set,"
                    "please make sure providing the full Gram matrix"
                    "as X_test."
                )
                # obtain train and test dimensions
                n_train = self.y_train.shape[0]
                n_test = X.shape[0] - n_train
                self.K_test = X[-n_test:, -n_test:]
                self.K_testtrain = X[n_train:, :n_train]
            else:
                raise ValueError("Unknown quantum kernel: {}".format(self._quantum_kernel))
        elif isinstance(self._quantum_kernel, KernelMatrixBase):
            self.K_test = self._quantum_kernel.evaluate(x=X)
            self.K_testtrain = self._quantum_kernel.evaluate(x=X, y=self.X_train)
        else:
            raise ValueError(
                "Unknown type of quantum kernel: {}".format(type(self._quantum_kernel))
            )

        if self.full_regularization:
            self.K_train, self.K_testtrain, self.K_test = regularize_full_kernel(
                self.K_train, self.K_testtrain, self.K_test
            )

        self.K_train += self.sigma * np.identity(self.K_train.shape[0])

        try:
            self._LU, self._piv = lu_factor(self.K_train)
        except np.linalg.LinAlgError:
            self.K_train += 1e-8 * np.identity(self.K_train.shape[0])
            self._LU, self._piv = lu_factor(self.K_train)
        self._alpha = lu_solve((self._LU, self._piv), self.y_train)
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
        v = lu_solve((self._LU, self._piv), self.K_testtrain.T)
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
            "full_regularization": self.full_regularization,
        }
        if deep and isinstance(self._quantum_kernel, KernelMatrixBase):
            params.update(self._quantum_kernel.get_params(deep=deep))
        return params

    def set_params(self, **params) -> None:
        """
        Sets value of the encoding circuit hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. ``num_qubits=2``.
        """
        valid_params = self.get_params()
        for key in params.keys():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

        # Set parameters of the QGPR
        self_params = self.get_params(deep=False).keys() & params.keys()
        for key in self_params:
            try:
                setattr(self, key, params[key])
            except AttributeError:
                setattr(self, "_" + key, params[key])

        if isinstance(self._quantum_kernel, KernelMatrixBase):
            # Set parameters of the Quantum Kernel and its underlying objects
            quantum_kernel_params = self._quantum_kernel.get_params().keys() & params.keys()
            if quantum_kernel_params:
                self._quantum_kernel.set_params(
                    **{key: params[key] for key in quantum_kernel_params}
                )
        return self
