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
from ...expectation_operator import SinglePauli
from ...expectation_operator.expectation_operator_base import ExpectationOperatorBase


class OuterKernelBase:
    """
    Base Class for creating outer kernels for the projected quantum kernel
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
                kernel (sklearn.gaussian_process.kernels): Sklearn kernel
                **kwarg: Arguments for the sklearn kernel parameters
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
    r"""Projected Quantum Kernel for Quantum Kernel Algorithms

    The Projected Quantum Kernel embeds classical data into a quantum Hilbert space first and
    than projects down into a real space by measurements. The real space is than used to
    evaluate a classical kernel.

    The projection is done by evaluating the expectation values of the feature map with respect
    to given Pauli operators. This is achieved by supplying a list of
    :class:`squlearn.expectation_operator` objects to the Projected Quantum Kernel.
    The expectation values are than used as features for the classical kernel, for which
    the different implementations of sklearn's kernels can be used.

    The implementation is based on the paper https://doi.org/10.1038/s41467-021-22539-9

    As defaults, a Gaussian outer kernel and the expectation value of all three Pauli matrices
    :math:`\{\hat{X},\hat{Y},\hat{Z}\}` are computed for every qubit.


    Args:
        feature_map (FeatureMapBase): Feature map that is evaluated
        executor (Executor): Executor object
        measurement (Union[str, ExpectationOperatorBase, list]): Expectation values that are
            computed from the feature map. Either an operator, a list of operators or one of the
            string values: ``X``,``Y``,``Z``, or ``XYZ``
        outer_kernel (Union[str, OuterKernelBase]): OuterKernel that is applied to the expectation
            values. Possible string values are: ``Gaussian``, ``Matern``, ``ExpSineSquared``,
            ``RationalQuadratic``, ``DotProduct``, ``PairwiseKernel``
        initial_parameters (np.ndarray): Initial parameters of the feature map and the
            operator (if parameterized)

    Attributes:
        num_qubits (int): Number of qubits of the feature map and the operators
        num_features (int): Number of features of the feature map
        num_parameters (int): Number of trainable parameters of the feature map
        feature_map (FeatureMapBase): Feature map that is evaluated
        measurement (Union[str, ExpectationOperatorBase, list]): Measurements that are
            performed on the feature map
        outer_kernel (Union[str, OuterKernelBase]): OuterKernel that is applied to the expectation
            values
        num_hyper_parameters (int): Number of hyper parameters of the outer kernel
        name_hyper_parameters (List[str]): Names of the hyper parameters of the outer kernel
        parameters (np.ndarray): Parameters of the feature map and the
            operator (if parameterized)

    Outer Kernels are implemented as follows:
    =========================================

    :math:`d(\cdot,\cdot)` is the Euclidean distance between two vectors.

    Gaussian:
    ---------
    .. math::
        k(x_i, x_j) = \text{exp}\left(-\gamma |(QNN(x_i)- QNN(x_j)|^2 \right)

    *Keyword Args:*

    :gamma (float): hyperparameter :math:`\gamma` of the Gaussian kernel

    Matern:
    -------
    .. math::
         k(x_i, x_j) =  \frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg(
         \!\frac{\sqrt{2\nu}}{l} d(QNN(x_i) , QNN(x_j))\!
         \Bigg)^\nu K_\nu\Bigg(
         \!\frac{\sqrt{2\nu}}{l} d(QNN(x_i) , QNN(x_j))\!\Bigg)

    *Keyword Args:*

    :nu (float): hyperparameter :math:`\nu` of the Matern kernel (Typically ``0.5``, ``1.5``
                 or ``2.5``)
    :length_scale (float): hyperparameter :math:`l` of the Matern kernel

    ExpSineSquared:
    ---------------
    .. math::
        k(x_i, x_j) = \text{exp}\left(-
        \frac{ 2\sin^2(\pi d(QNN(x_i), QNN(x_j))/p) }{ l^ 2} \right)

    *Keyword Args:*

    :periodicity (float): hyperparameter :math:`p` of the ExpSineSquared kernel
    :length_scale (float): hyperparameter :math:`l` of the ExpSineSquared kernel

    RationalQuadratic:
    ------------------
    .. math::
        k(x_i, x_j) = \left(
        1 + \frac{d(QNN(x_i), QNN(x_j))^2 }{ 2\alpha  l^2}\right)^{-\alpha}

    *Keyword Args:*

    :alpha (float): hyperparameter :math:`\alpha` of the RationalQuadratic kernel
    :length_scale (float): hyperparameter :math:`l` of the RationalQuadratic kernel

    DotProduct:
    -----------
    .. math::
        k(x_i, x_j) = \sigma_0 ^ 2 + x_i \cdot x_j

    *Keyword Args:*

    :sigma_0 (float): hyperparameter :math:`\sigma_0` of the DotProduct kernel

    PairwiseKernel:
    ---------------

    sklearn's PairwiseKernel is used.

    *Keyword Args:*

    :gamma (float): Hyperparameter gamma of the PairwiseKernel kernel, specified by the metric
    :metric (str): Metric of the PairwiseKernel kernel, can be ``linear``, ``additive_chi2``,
              ``chi2``, ``poly``, ``polynomial``, ``rbf``, ``laplacian``, ``sigmoid``, ``cosine``

    See Also:
        * Quantum Fidelity Kernel: :class:`squlearn.kernel.matrix.FidelityKernel`
        * `sklean kernels <https://scikit-learn.org/stable/modules/gaussian_process.html#gp-kernels>`_

    References:
        * Huang, HY., Broughton, M., Mohseni, M. et al.
          Power of data in quantum machine learning. Nat Commun 12, 2631 (2021).
          https://doi.org/10.1038/s41467-021-22539-9

    **Example: Calculate a kernel matrix with the Projected Quantum Kernel**

    .. code-block:: python

       import numpy as np
       from squlearn.feature_map import ChebyshevTower
       from squlearn.kernel.matrix import ProjectedQuantumKernel
       from squlearn.util import Executor

       fm = ChebyshevTower(num_qubits=4, num_features=1, num_chebyshev=4)
       kernel = ProjectedQuantumKernel(feature_map=fm, executor=Executor("statevector_simulator"))
       x = np.random.rand(10)
       kernel_matrix = kernel.evaluate(x.reshape(-1, 1), x.reshape(-1, 1))

    **Example: Change measurement and outer kernel**

    .. code-block:: python

       import numpy as np
       from squlearn.feature_map import ChebyshevTower
       from squlearn.kernel.matrix import ProjectedQuantumKernel
       from squlearn.util import Executor
       from squlearn.expectation_operator import CustomExpectationOperator
       from squlearn.kernel.ml import QKRR

       fm = ChebyshevTower(num_qubits=4, num_features=1, num_chebyshev=4)

       # Create custom expectation operators
       measuments = []
       measuments.append(CustomExpectationOperator(4,"ZZZZ"))
       measuments.append(CustomExpectationOperator(4,"YYYY"))
       measuments.append(CustomExpectationOperator(4,"XXXX"))

       # Use Matern Outer kernel with nu=0.5 as a outer kernel hyperparameter
       kernel = ProjectedQuantumKernel(feature_map=fm,
                                       executor=Executor("statevector_simulator"),
                                       measurement=measuments,
                                       outer_kernel="matern",
                                       nu=0.5)
       ml_method = QKRR(quantum_kernel=kernel)

    Methods:
    --------
    """

    def __init__(
        self,
        feature_map: FeatureMapBase,
        executor: Executor,
        measurement: Union[str, ExpectationOperatorBase, list] = "XYZ",
        outer_kernel: Union[str, OuterKernelBase] = "gaussian",
        initial_parameters: Union[np.ndarray, None] = None,
        **kwargs,
    ) -> None:
        super().__init__(feature_map, executor, initial_parameters)

        self._measurement_input = measurement

        # Set-up measurement operator
        if isinstance(measurement, str):
            if measurement == "Z":
                # Measure Z at all qubits
                self._measurement = []
                for i in range(self.num_qubits):
                    self._measurement.append(SinglePauli(self.num_qubits, i, op_str="Z"))
            elif measurement == "X":
                # Measure Z at all qubits
                self._measurement = []
                for i in range(self.num_qubits):
                    self._measurement.append(SinglePauli(self.num_qubits, i, op_str="X"))
            elif measurement == "Y":
                # Measure Z at all qubits
                self._measurement = []
                for i in range(self.num_qubits):
                    self._measurement.append(SinglePauli(self.num_qubits, i, op_str="Y"))
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
            if len(self._parameters) != self.num_parameters:
                raise ValueError(
                    "Number of inital parameters is wrong, expected number: {}".format(
                        self.num_parameters
                    )
                )

    @property
    def num_features(self) -> int:
        """Feature dimension of the feature map"""
        return self._qnn.num_features

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters of the feature map"""
        return self._qnn.num_parameters + self._qnn.num_parameters_operator

    @property
    def measurement(self):
        """Measurement operator of the Projected Quantum Kernel"""
        return self._measurement

    @property
    def outer_kernel(self):
        """Outer kernel class of the Projected Quantum Kernel"""
        return self._outer_kernel

    def evaluate_qnn(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the QNN for the given data x.

        Args:
            x (np.ndarray): Data points x
        Returns:
            The evaluated output of the QNN as numpy array
        """

        # Copy parameters in QNN form
        if self._parameters is None and self.num_parameters == 0:
            self._parameters = []
        if self._parameters is None:
            raise ValueError("Parameters have not been set yet!")
        param = self._parameters[: self._qnn.num_parameters]
        param_op = self._parameters[self._qnn.num_parameters :]
        # Evaluate and return
        return self._qnn.evaluate_f(x, param, param_op)

    def evaluate(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Evaluates the Projected Quantum Kernel for the given data points x and y.

        Args:
            x (np.ndarray): Data points x
            y (np.ndarray): Data points y, if None y = x is used

        Returns:
            The evaluated projected quantum kernel as numpy array
        """
        if self._parameters is None and self.num_parameters == 0:
            self._parameters = []

        if self._parameters is None:
            raise ValueError("Parameters have not been set yet!")

        return self._outer_kernel(self._qnn, self._parameters, x, y)

    def get_params(self) -> dict:
        """Returns the hyper parameters of the projected kernel"""
        params = {}
        params.update(self._outer_kernel.get_params())
        params.update(self._qnn.get_params())
        params["measurement"] = self._measurement_input
        return params

    def set_params(self, **params):

        num_parameters_backup = self.num_parameters
        parameters_backup = self._parameters

        """Sets the hyper parameters of the outer kernel"""
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s. "
                    "Check the list of available parameters "
                    "with `QNN.get_params()`." % key
                )

        # Set QNN parameters
        dict_qnn = {}
        for key, value in params.items():
            if key in self._qnn.get_params():
                dict_qnn[key] = value
        if len(dict_qnn) > 0:
            self.pqc.set_params(**dict_qnn)

        # Set outer kernel parameters
        dict_outer_kernel = {}
        for key, value in params.items():
            if key in self._outer_kernel.get_params():
                dict_outer_kernel[key] = value
        if len(dict_outer_kernel) > 0:
            self._outer_kernel.set_params(**dict_outer_kernel)

        # Set measurement -> re-initialize Projected Quantum Kernel module
        if "measurement" in params:
            self._measurement_input = params["measurement"]
            self.__init__(
                self._feature_map,
                self._executor,
                self._measurement_input,
                self._outer_kernel,
                self._parameters,
            )

        self._parameters = None
        if self.num_parameters == num_parameters_backup:
            self._parameters = parameters_backup


    @property
    def num_hyper_parameters(self) -> int:
        """The number of hyper-parameters of the outer kernel"""
        return self._outer_kernel.num_hyper_parameters

    @property
    def name_hyper_parameters(self) -> List[str]:
        """The names of the hyper-parameters of the outer kernel"""
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
