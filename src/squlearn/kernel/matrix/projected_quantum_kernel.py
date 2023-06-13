"""Projected Quantum Kernel class"""

from typing import Union, List
import numpy as np
from abc import abstractmethod

from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    ExpSineSquared,
    RationalQuadratic,
    DotProduct,
    PairwiseKernel,
)

from sklearn.gaussian_process.kernels import Kernel as SklearnKernel

from .kernel_matrix_base import KernelMatrixBase
from ...feature_map.feature_map_base import FeatureMapBase
from ...util import Executor
from ...qnn.qnn import QNN
from ...expectation_operator import SinglePauli, ExpectationOperatorBase


class OuterKernelBase:
    """
    Class for creating outer kernels for the projected quantum kernel
    """

    def __init__(self):
        self._num_hyper_parameters = 0
        self._name_hyper_parameters = []

    @abstractmethod
    def __call__(
        self, qnn: QNN, parameters: np.ndarray, x: np.ndarray, y: np.ndarray = None
    ) -> np.ndarray:
        """
        Args:
            qnn: QNN object
            parameters: parameters of the QNN
            x: first input
            y: second optional input

        Returns:
            Evaluated projected kernel matrix
        """
        raise NotImplementedError()

    @abstractmethod
    def get_params(self) -> dict:
        """Returns the hyper parameters of the outer kernel"""
        raise NotImplementedError()

    @abstractmethod
    def set_params(self, **kwarg):
        """Sets the hyper parameters of the outer kernel"""
        raise NotImplementedError()

    @property
    def num_hyper_parameters(self) -> int:
        """Returns the number of hyper parameters of the outer kernel"""
        return self._num_hyper_parameters

    @property
    def name_hyper_parameters(self) -> List[str]:
        """Returns the names of the hyper parameters of the outer kernel"""
        return self._name_hyper_parameters

    @classmethod
    def from_sklearn_kernel(cls, kernel: SklearnKernel, **kwarg):
        """Converts a sklearn kernel into a squlearn kernel

        Args:
            kernel: sklearn kernel
            kwarg: arguments for the sklearn kernel parameters
        """

        class SklearnOuterKernel(BaseException):
            """
            Class for creating outer kernels for the projected quantum kernel from sklearn kernels

            Args:
                kernel: sklearn kernel
                kwarg: arguments for the sklearn kernel parameters
            """

            def __init__(self, kernel: SklearnKernel, **kwarg):
                super().__init__()
                self._kernel = kernel(**kwarg)
                self._name_hyper_parameters = [p.name for p in self._kernel.hyperparameters]
                self._num_hyper_parameters = len(self._name_hyper_parameters)

            def __call__(
                self, qnn: QNN, parameters: np.ndarray, x: np.ndarray, y: np.ndarray = None
            ) -> np.ndarray:
                """Evaluates the outer kernel

                Args:
                    qnn: QNN object
                    parameters: parameters of the QNN
                    x: first input
                    y: second optional input
                """
                # Evaluate QNN
                param = parameters[: qnn.num_parameters]
                param_op = parameters[qnn.num_parameters :]
                x_result = qnn.evaluate_f(x, param, param_op)
                if y is not None:
                    y_result = qnn.evaluate_f(y, param, param_op)
                else:
                    y_result = None
                # Evaluate kernel
                return self._kernel(x_result, y_result)

            def get_params(self) -> dict:
                """Returns the hyper parameters of the outer kernel as a dictionary."""
                return self._kernel.get_params()

            def set_params(self, **kwarg):
                """Sets the hyper parameters of the outer kernel."""
                self._kernel.set_params(**kwarg)

        return SklearnOuterKernel(kernel, **kwarg)


class ProjectedQuantumKernel(KernelMatrixBase):

    """Projected Quantum Kernel

    The projected quantum kernel embeds classical data into a quantum Hilbert space and
    than projects down into a real space by measurements. The kernel is than evaluated in the
    real space. The implementation is based on the paper https://doi.org/10.1038/s41467-021-22539-9

    Args:
        feature_map (FeatureMapBase): PQC feature map
        executor (Executor): Executor object
        measurement (Union[str, ExpectationOperatorBase, list]): Measurements that are
            performed on the PQC. Possible string values: "Z", "XYZ"
        outer_kernel (Union[str, OuterKernelBase]): OuterKernel that is applied to the PQC output.
            Possible string values are: "Gaussian", "Matern", "ExpSineSquared",
            "RationalQuadratic", "DotProduct", "PairwiseKernel"
        initial_parameters (np.ndarray): initial parameters of the QNN

    Outer Kernels that are implemented:

    Gaussian:
    ---------
    .. math::
        k(x_i, x_j) = \text{exp}\left(-\\gamma |(QNN(x_i)- QNN(x_j)|^2 \right)

    Args:
        gamma (float): hyperparameter :math:`\\gamma` of the Gaussian kernel

    Matern:
    -------
    .. math::
         k(x_i, x_j) =  \\frac{1}{\\Gamma(\\nu)2^{\\nu-1}}\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(QNN(x_i) , QNN(x_j) )
         \\Bigg)^\\nu K_\\nu\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(QNN(x_i) , QNN(x_j) )\\Bigg)

    Args:
        nu (float): hyperparameter :math:`\\nu` of the Matern kernel (Typically 0.5, 1.5 or 2.5)
        length_scale (float): hyperparameter :math:`l` of the Matern kernel

    ExpSineSquared:
    ---------------
    .. math::
        k(x_i, x_j) = \text{exp}\left(-
        \frac{ 2\sin^2(\pi d(QNN(x_i), QNN(x_j))/p) }{ l^ 2} \right)

    Args:
        periodicity (float): hyperparameter :math:`p` of the ExpSineSquared kernel
        length_scale (float): hyperparameter :math:`l` of the ExpSineSquared kernel

    RationalQuadratic:
    ------------------
    .. math::
        k(x_i, x_j) = \\left(
        1 + \\frac{d(QNN(x_i), QNN(x_j))^2 }{ 2\\alpha  l^2}\\right)^{-\\alpha}s

    Args:
        alpha (float): hyperparameter :math:`\\alpha` of the RationalQuadratic kernel
        length_scale (float): hyperparameter :math:`l` of the RationalQuadratic kernel

    DotProduct:
    -----------
    .. math::
        k(x_i, x_j) = \sigma_0 ^ 2 + x_i \cdot x_j

    Args:
        sigma_0 (float): hyperparameter :math:`\sigma_0` of the DotProduct kernel

    PairwiseKernel:
    ---------------
    Args:
        gamma (float): Hyperparameter gamma of the PairwiseKernel kernel, specified by the metric
        metric (str): Metric of the PairwiseKernel kernel, can be "linear", "additive_chi2",
              "chi2", "poly", "polynomial", "rbf", "laplacian", "sigmoid", "cosine"

    """

    def __init__(
        self,
        feature_map: FeatureMapBase,
        executor: Executor,
        measurement: Union[str, ExpectationOperatorBase, list] = "XYZ",
        outer_kernel: Union[str, OuterKernelBase] = "gaussian",
        initial_parameters: np.ndarray = None,
        **kwargs,
    ) -> None:
        super().__init__(feature_map, executor, initial_parameters)

        # Set-up measurement operator
        if isinstance(measurement, str):
            if measurement == "Z":
                # Measure Z at all qubits
                self._measurement = []
                for i in range(self.num_qubits):
                    self._measurement.append(SinglePauli(self.num_qubits, i, op_str="Z"))
            elif measurement == "XYZ":
                # Measure X, Y, and Z at all qubits
                self._measurement = []
                for i in range(self.num_qubits):
                    self._measurement.append(SinglePauli(self.num_qubits, i, op_str="X"))
                    self._measurement.append(SinglePauli(self.num_qubits, i, op_str="Y"))
                    self._measurement.append(SinglePauli(self.num_qubits, i, op_str="Z"))
            else:
                raise ValueError("Unknown measurement string: {}".format(measurement))
        elif isinstance(measurement, ExpectationOperatorBase) or isinstance(measurement, list):
            self._measurement = measurement
        else:
            raise ValueError("Unknown type of measurement: {}".format(type(measurement)))

        # Set-up of the QNN
        self._qnn = QNN(self._feature_map, self._measurement, executor)
        self._num_param = self._qnn.num_parameters
        self._num_param_op = self._qnn.num_parameters_operator
        self._num_parameters = self._num_param + self._num_param_op

        # Set-up of the outer kernel
        if isinstance(outer_kernel, str):
            if outer_kernel.lower() == "gaussian":
                self._outer_kernel = GaussianOuterKernel(**kwargs)
            elif outer_kernel.lower() == "matern":
                self._outer_kernel = OuterKernelBase.from_sklearn_kernel(Matern, **kwargs)
            elif outer_kernel.lower() == "expsinesquared":
                self._outer_kernel = OuterKernelBase.from_sklearn_kernel(ExpSineSquared, **kwargs)
            elif outer_kernel.lower() == "rationalquadratic":
                self._outer_kernel = OuterKernelBase.from_sklearn_kernel(
                    RationalQuadratic, **kwargs
                )
            elif outer_kernel.lower() == "dotproduct":
                self._outer_kernel = OuterKernelBase.from_sklearn_kernel(DotProduct, **kwargs)
            elif outer_kernel.lower() == "pairwisekernel":
                self._outer_kernel = OuterKernelBase.from_sklearn_kernel(PairwiseKernel, **kwargs)
            else:
                raise ValueError("Unknown outer kernel: {}".format(outer_kernel))
        elif isinstance(outer_kernel, OuterKernelBase):
            self._outer_kernel = outer_kernel
        else:
            raise ValueError("Unknown type of outer kernel: {}".format(type(outer_kernel)))

        # Check if the number of parameters is correct
        if self._parameters is not None:
            if len(self._parameters) != self._num_parameters:
                raise ValueError(
                    "Number of inital parameters is wrong, expected number: {}".format(
                        self._num_parameters
                    )
                )

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def num_parameters(self) -> int:
        return self._num_parameters

    @property
    def measurement(self):
        return self._measurement

    @property
    def outer_kernel(self):
        return self._outer_kernel

    def evaluate_qnn(self, x: np.ndarray) -> np.ndarray:
        # Copy parameters in QNN form
        if self._parameters is None:
            raise ValueError("Parameters have not been set yet!")
        param = self._parameters[: self._qnn.num_parameters]
        param_op = self._parameters[self._qnn.num_parameters :]
        return self._qnn.evaluate_f(x, param, param_op)

    def evaluate(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Evaluates the Projected Quantum Kernel for the given data points x and y.

        Args:
            x (np.ndarray): Data points x
            y (np.ndarray): Data points y, if None y = x is used

        Returns:
            The evaluated projected quantum kernel as numpy array
        """
        if self._parameters is None:
            raise ValueError("Parameters have not been set yet!")

        return self._outer_kernel(self._qnn, self._parameters, x, y)

    def get_params(self) -> dict:
        """Returns the hyper parameters of the outer kernel"""
        return self._outer_kernel.get_params()

    def set_params(self, **kwarg):
        """Sets the hyper parameters of the outer kernel"""
        self._outer_kernel.set_params(**kwarg)

    @property
    def num_hyper_parameters(self) -> int:
        """Returns the number of hyper parameters of the outer kernel"""
        return self._outer_kernel.num_hyper_parameters

    @property
    def name_hyper_parameters(self) -> List[str]:
        """Returns the names of the hyper parameters of the outer kernel"""
        return self._outer_kernel.name_hyper_parameters


class GaussianOuterKernel(OuterKernelBase):
    """
    Implementation of the Gaussian outer kernel:

    .. math::
        k(x_i, x_j) = \text{exp}\left(-\\gamma |(QNN(x_i)- QNN(x_j)|^2 \right)

    Args:
        gamma (float): hyperparameter :math:`\\gamma` of the Gaussian kernel
    """

    def __init__(self, gamma=1.0):
        super().__init__()
        self._gamma = gamma
        self._num_hyper_parameters = 1
        self._name_hyper_parameters = ["gamma"]

    def __call__(
        self, qnn: QNN, parameters: np.ndarray, x: np.ndarray, y: np.ndarray = None
    ) -> np.ndarray:
        """Evaluates the QNN and returns the Gaussian projected kernel

        Args:
            qnn (QNN): QNN to be evaluated
            parameters (np.ndarray): parameters of the QNN
            x (np.ndarray): input data
            y (np.ndarray): second optional input data

        Returns:
            np.ndarray: Gaussian projected kernel
        """

        # Evaluate QNN
        param = parameters[: qnn.num_parameters]
        param_op = parameters[qnn.num_parameters :]
        x_result = qnn.evaluate_f(x, param, param_op)
        if y is not None:
            y_result = qnn.evaluate_f(y, param, param_op)
        else:
            y_result = None

        return RBF(length_scale=1.0 / np.sqrt(2.0 * self._gamma))(x_result, y_result)

    def get_params(self) -> dict:
        """Returns the hyper parameters of the outer kernel."""
        return {"gamma": self._gamma}

    def set_params(self, gamma) -> None:
        """Sets the hyper parameters of the outer kernel."""
        self._gamma = gamma
