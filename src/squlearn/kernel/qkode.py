"""Quantum Kernel ODE"""

from functools import partial
from typing import Union
import warnings

import numpy as np
from packaging import version
from sklearn import __version__

from squlearn.util.data_preprocessing import extract_num_features

if version.parse(__version__) >= version.parse("1.6"):
    from sklearn.utils.validation import validate_data
else:

    def validate_data(self, *args, **kwargs):
        return self._validate_data(*args, **kwargs)


from .lowlevel_kernel.kernel_matrix_base import KernelMatrixBase
from .qkrr import QKRR
from .loss.kernel_loss_base import KernelLossBase
from ..optimizers.optimizer_base import OptimizerBase


class QKODE(QKRR):
    r"""
    Quantum Kernel Ordinary Differential Equation (QKODE) solver.

    This class implements a quantum kernel-based solver for ordinary differential equations
    (ODEs) using the mixed model regression method as described in Ref. [1].

    Args:
        quantum_kernel (Union[KernelMatrixBase, str]): Quantum kernel to be used in the model.
            If set to "precomputed",
            the derivatives of the kernel matrix have to be provided.
        loss (KernelLossBase): Loss function to be used for training the model.
        optimizer (OptimizerBase): Optimizer to be used for minimizing the loss function.
        alpha_seed (int, default=0): Seed for random initialization of dual coefficients.
        k_train (np.ndarray): Precomputed training kernel matrix of shape (n_train, n_train).
            Required if quantum_kernel is "precomputed".
        dkdx_train (np.ndarray): Precomputed first derivatives of the training kernel matrix.
            Required if quantum_kernel is "precomputed".
        dkdxdx_train (np.ndarray): Precomputed second derivatives of the training kernel matrix.
            Required if quantum_kernel is "precomputed" and the ODE is of order 2.
        **kwargs: Additional keyword arguments to be passed to the base class.

    Attributes:
    -----------
        dual_coeff (np.ndarray) :
            Array containing the weight vector in kernel space.
        k_train (np.ndarray) :
            Training kernel matrix of shape (n_train, n_train) which is available after calling the
            fit procedure.
        k_testtrain (np.ndarray) :
            Kernel matrix of shape (n_test, n_train) which is evaluated at the predict step.

    See Also:
        squlearn.kernel.loss.ODELoss : Loss function for ODEs.

    References
    ----------
        [1]: A. Paine et al., "Quantum kernel methods for solving regression problems and
        differential equations", Phys. Rev. A 107, 032428


    Methods:
    --------
    """

    def __init__(
        self,
        quantum_kernel: Union[KernelMatrixBase, str],
        loss: KernelLossBase,
        optimizer: OptimizerBase,
        alpha_seed: int = 0,
        k_train: np.ndarray = None,
        dkdx_train: np.ndarray = None,
        dkdxdx_train: np.ndarray = None,
        **kwargs,
    ) -> None:
        super().__init__(quantum_kernel=quantum_kernel, alpha=None, **kwargs)
        self._loss = loss
        self._loss.set_quantum_kernel(quantum_kernel)
        self._optimizer = optimizer
        self.alpha_seed = alpha_seed
        self.k_train = k_train
        self.dkdx_train = dkdx_train
        self.dkdxdx_train = dkdxdx_train

        if quantum_kernel == "precomputed":
            if k_train is None or dkdx_train is None:
                raise ValueError(
                    "If quantum_kernel is 'precomputed', the training kernel matrix and its first"
                    " derivatives have to be provided via k_train and dkdx_train."
                )
            if loss.order_of_ode == 2 and dkdxdx_train is None:
                raise ValueError(
                    "If quantum_kernel is 'precomputed' and the ODE is of order 2, the second "
                    "derivatives of the training kernel matrix have to be provided via "
                    "dkdxdx_train."
                )
        elif not isinstance(quantum_kernel, KernelMatrixBase):
            raise ValueError("Unknown type of quantum kernel: {}".format(type(quantum_kernel)))

    def fit(self, X, y):
        """
        Fit the Quantum Kernel ODE model.

        Args:
            X (np.ndarray) : Samples of data of shape (n_samples, n_features) used for fitting the
                QKODE model.
            y (np.ndarray) : Labels of shape (n_samples,) used for fitting the QKODE model.

        """
        X, y = validate_data(
            self, X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )
        self.X_train = X

        # set up kernel matrix
        if isinstance(self._quantum_kernel, str):
            if self._quantum_kernel == "precomputed":
                # if kernel is precomputed, validate shape of kernel matrix
                K, y = validate_data(
                    self,
                    self.k_train,
                    y,
                    accept_sparse=("csr", "csc"),
                    multi_output=True,
                    y_numeric=True,
                )
                self.k_train = K
            else:
                raise ValueError("Unknown quantum kernel: {}".format(self._quantum_kernel))
        elif isinstance(self._quantum_kernel, KernelMatrixBase):
            # initialize the kernel with the known feature vector
            num_features = extract_num_features(X)
            self._quantum_kernel._initialize_kernel(num_features=num_features)

            # check if quantum kernel is trainable
            if self._quantum_kernel.is_trainable:
                warnings.warn(
                    "The Quantum Kernel is trainable but training the parameters of the kernel is"
                    " not supported yet. Setting random parameters."
                )

            self.k_train = self._quantum_kernel.evaluate_derivatives(self.X_train, values="K")
            self.dkdx_train = self._quantum_kernel.evaluate_derivatives(
                self.X_train, values="dKdx"
            )
            if self._loss.order_of_ode == 2:
                self.dkdxdx_train = self._quantum_kernel.evaluate_derivatives(
                    self.X_train, self.X_train, values="dKdxdx"
                )

        else:
            raise ValueError(
                "Unknown type of quantum kernel: {}".format(type(self._quantum_kernel))
            )

        random_device = np.random.RandomState(seed=self.alpha_seed)
        alpha_ini = random_device.rand(len(y) + 1)

        # pass self into the loss function
        loss_function = partial(
            self._loss.compute,
            data=X,
            kernel_tensor=[self.k_train, self.dkdx_train, self.dkdxdx_train],
        )
        opt_result = self._optimizer.minimize(fun=loss_function, x0=alpha_ini)
        self.dual_coeff_ = opt_result.x
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Quantum Kernel Ridge model.

        Args:
            X (np.ndarray) : Samples of data of shape (n_samples, n_features) on which QKRR
                model makes predictions. If quantum_kernel == "precomputed" this is instead a
                precomputed (test-train) kernel matrix of shape (n_samples, n_samples_fitted),
                where n_samples_fitted is the number of samples used in the fitting.

        Returns:
            np.ndarray :
                Returns predicted labels (at X) of shape (n_samples,)
        """
        if self.k_train is None:
            raise ValueError("The fit() method has to be called beforehand.")

        X = validate_data(self, X, accept_sparse=("csr", "csc"), reset=False)

        if isinstance(self._quantum_kernel, str):
            if self._quantum_kernel == "precomputed":
                self.k_testtrain = X
        elif isinstance(self._quantum_kernel, KernelMatrixBase):
            self.k_testtrain = self._quantum_kernel.evaluate(x=X, y=self.X_train)
        else:
            raise ValueError(
                "Unknown type of quantum kernel: {}".format(type(self._quantum_kernel))
            )

        prediction = np.dot(self.k_testtrain, self.dual_coeff_[1:]) + self.dual_coeff_[0]
        return prediction
