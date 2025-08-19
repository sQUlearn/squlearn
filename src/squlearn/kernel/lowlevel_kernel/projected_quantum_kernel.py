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
from ...encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ...util import Executor
from ...util.data_preprocessing import to_tuple


from ...qnn.lowlevel_qnn import LowLevelQNN
from ...qnn.lowlevel_qnn.lowlevel_qnn_base import LowLevelQNNBase

from ...observables import SinglePauli
from ...observables.observable_base import ObservableBase


class OuterKernelBase:
    """
    Base Class for creating outer kernels for the projected quantum kernel
    """

    def __init__(self):
        self._num_hyper_parameters = 0
        self._name_hyper_parameters = []

    @abstractmethod
    def __call__(
        self, qnn: LowLevelQNNBase, parameters: np.ndarray, x: np.ndarray, y: np.ndarray = None
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
    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the outer kernel.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_params(self, **params):
        """
        Sets value of the outer kernel hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. ``num_qubits=2``
        """
        raise NotImplementedError()

    def dKdx(
        self,
        qnn: LowLevelQNNBase,
        parameters: np.ndarray,
        x: np.ndarray,
        y: np.ndarray = None,
        with_respect_to: str = "dx",
    ) -> np.ndarray:
        """
        Implements the analytical derivative of the outer kernel with respect to x.

        Args:
            qnn (QNN): QNN to be evaluated
            parameters (np.ndarray): parameters of the QNN
            x (np.ndarray): input data (n, num_features)
            y (np.ndarray): second optional input data (n, num_features)

        Returns:
            np.ndarray: derivative of the outer projected kernel of shape (len(X), len(Y), num_qubits*len(measurement))
        """

        raise NotImplementedError("Kernel derivatives are not implement for the outer kernel")

    def dKdxdx(
        self, qnn: LowLevelQNNBase, parameters: np.ndarray, x: np.ndarray, y: np.ndarray = None
    ) -> np.ndarray:
        """
        Implements the analytical derivative of the outer kernel with respect to x and x.

        Args:
            qnn (QNN): QNN to be evaluated
            parameters (np.ndarray): parameters of the QNN
            x (np.ndarray): input data
            y (np.ndarray): second optional input data

        Returns:
            np.ndarray: derivative dKdxdx of the outer projected kernel shape (len(X), len(Y), num_qubits*len(measurement))
        """
        raise NotImplementedError("Kernel derivatives are not implement for the outer kernel")

    def dKdxdy(
        self, qnn: LowLevelQNNBase, parameters: np.ndarray, x: np.ndarray, y: np.ndarray = None
    ) -> np.ndarray:
        """
        Implements the analytical derivative of the outer kernel with respect to x and y.

        Args:
            qnn (QNN): QNN to be evaluated
            parameters (np.ndarray): parameters of the QNN
            x (np.ndarray): input data
            y (np.ndarray): second optional input data

        Returns:
            np.ndarray: derivative dKdxdy of the outer projected kernel shape (len(X), len(Y), num_qubits*len(measurement), 1)
        """
        raise NotImplementedError("Kernel derivatives are not implement for the outer kernel")

    @property
    def num_hyper_parameters(self) -> int:
        """Returns the number of hyper parameters of the outer kernel"""
        return self._num_hyper_parameters

    @property
    def name_hyper_parameters(self) -> List[str]:
        """Returns the names of the hyper parameters of the outer kernel"""
        return self._name_hyper_parameters

    @classmethod
    def from_sklearn_kernel(cls, kernel: str, **kwarg):
        """Converts a scikit-learn kernel into a squlearn kernel

        Args:
            kernel: scikit-learn kernel
            kwarg: arguments for the scikit-learn kernel parameters
        """

        class SklearnOuterKernel(BaseException):
            """
            Class for creating outer kernels for the projected quantum kernel from scikit-learn kernels

            Args:
                kernel (:py:mod:`sklearn.gaussian_process.kernels`): Scikit-learn kernel
                **kwarg: Arguments for the scikit-learn kernel parameters
            """

            def __init__(self, kernel: str, **kwarg):
                super().__init__()

                if kernel.lower() == "matern":
                    self._kernel = Matern(**kwarg)
                elif kernel.lower() == "expsinesquared":
                    self._kernel = ExpSineSquared(**kwarg)
                elif kernel.lower() == "rationalquadratic":
                    self._kernel = RationalQuadratic(**kwarg)
                elif kernel.lower() == "dotproduct":
                    self._kernel = DotProduct(**kwarg)
                elif kernel.lower() == "pairwisekernel":
                    self._kernel = PairwiseKernel(**kwarg)
                else:
                    raise ValueError("Unknown scikit-learn kernel: {}".format(kernel))

                self._name_hyper_parameters = [p.name for p in self._kernel.hyperparameters]
                self._num_hyper_parameters = len(self._name_hyper_parameters)

            def __call__(
                self,
                qnn: LowLevelQNNBase,
                parameters: np.ndarray,
                x: np.ndarray,
                y: np.ndarray = None,
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
                x_result = qnn.evaluate(x, param, param_op, "f")["f"]
                if y is not None:
                    y_result = qnn.evaluate(y, param, param_op, "f")["f"]
                else:
                    y_result = None
                # Evaluate kernel
                return self._kernel(x_result, y_result)

            def get_params(self, deep: bool = True) -> dict:
                """
                Returns hyper-parameters and their values of the scikit-learn kernel.

                Args:
                    deep (bool): If True, also the parameters for
                                contained objects are returned (default=True).

                Return:
                    Dictionary with hyper-parameters and values.
                """
                return self._kernel.get_params(deep)

            def set_params(self, **params):
                """
                Sets value of the scikit-learn kernel.

                Args:
                    params: Hyper-parameters and their values
                """
                self._kernel.set_params(**params)

        return SklearnOuterKernel(kernel, **kwarg)


class ProjectedQuantumKernel(KernelMatrixBase):
    r"""Projected Quantum Kernel for Quantum Kernel Algorithms

    The Projected Quantum Kernel embeds classical data into a quantum Hilbert space first and
    than projects down into a real space by measurements. The real space is than used to
    evaluate a classical kernel.

    The projection is done by evaluating the expectation values of the encoding circuit with respect
    to given Pauli operators. This is achieved by supplying a list of
    :class:`squlearn.observable` objects to the Projected Quantum Kernel.
    The expectation values are than used as features for the classical kernel, for which
    the different implementations of scikit-learn's kernels can be used.

    The implementation is based on Ref. [1].

    As defaults, a Gaussian outer kernel and the expectation value of all three Pauli matrices
    :math:`\{\hat{X},\hat{Y},\hat{Z}\}` are computed for every qubit.


    Args:
        encoding_circuit (EncodingCircuitBase): Encoding circuit that is evaluated
        executor (Executor): Executor object
        measurement (Union[str, ObservableBase, list]): Expectation values that are
            computed from the encoding circuit. Either an operator, a list of operators or a
            combination of the string values ``X``,``Y``,``Z``, e.g. ``XYZ``
        outer_kernel (Union[str, OuterKernelBase]): OuterKernel that is applied to the expectation
            values. Possible string values are: ``Gaussian``, ``Matern``, ``ExpSineSquared``,
            ``RationalQuadratic``, ``DotProduct``, ``PairwiseKernel``
        initial_parameters (np.ndarray): Initial parameters of the encoding circuit and the
            operator (if parameterized)
        parameter_seed (Union[int, None], default=0):
            Seed for the random number generator for the parameter initialization, if
            initial_parameters is None.
        regularization  (Union[str, None], default=None):
            Option for choosing different regularization techniques (``"thresholding"`` or
            ``"tikhonov"``) after Ref. [2] for the training kernel matrix, prior to  solving the
            linear system in the ``fit()``-procedure.
        caching (bool, default=True): If True, the results of the low-level QNN are cached.


    Attributes:
    -----------

    Attributes:
        num_qubits (int): Number of qubits of the encoding circuit and the operators
        num_features (int): Number of features of the encoding circuit
        num_parameters (int): Number of trainable parameters of the encoding circuit
        encoding_circuit (EncodingCircuitBase): Encoding circuit that is evaluated
        measurement (Union[str, ObservableBase, list]): Measurements that are
            performed on the encoding circuit
        outer_kernel (Union[str, OuterKernelBase]): OuterKernel that is applied to the expectation
            values
        num_hyper_parameters (int): Number of hyper parameters of the outer kernel
        name_hyper_parameters (List[str]): Names of the hyper parameters of the outer kernel
        parameters (np.ndarray): Parameters of the encoding circuit and the
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

    scikit-learn's PairwiseKernel is used.

    *Keyword Args:*

    :gamma (float): Hyperparameter gamma of the PairwiseKernel kernel, specified by the metric
    :metric (str): Metric of the PairwiseKernel kernel, can be ``linear``, ``additive_chi2``,
              ``chi2``, ``poly``, ``polynomial``, ``rbf``, ``laplacian``, ``sigmoid``, ``cosine``

    See Also:
        * Quantum Fidelity Kernel: :class:`squlearn.kernel.lowlevel_kernel.FidelityKernel`
        * `sklean kernels <https://scikit-learn.org/stable/modules/gaussian_process.html#gp-kernels>`_

    References:
        [1] Huang, HY., Broughton, M., Mohseni, M. et al., "Power of data in quantum machine learning",
        `Nat Commun 12, 2631 (2021). <https://doi.org/10.1038/s41467-021-22539-9>`_

        [2] T. Hubregtsen et al., "Training Quantum Embedding Kernels on Near-Term Quantum Computers",
        `arXiv:2105.02276v1 (2021). <https://arxiv.org/abs/2105.02276>`_

    **Example: Calculate a kernel matrix with the Projected Quantum Kernel**

    .. jupyter-execute::

       import numpy as np
       from squlearn.encoding_circuit import ChebyshevTower
       from squlearn.kernel.lowlevel_kernel import ProjectedQuantumKernel
       from squlearn.util import Executor

       fm = ChebyshevTower(num_qubits=4, num_features=1, num_chebyshev=4)
       kernel = ProjectedQuantumKernel(encoding_circuit=fm, executor=Executor())
       x = np.random.rand(10)
       kernel_matrix = kernel.evaluate(x.reshape(-1, 1), x.reshape(-1, 1))
       print(kernel_matrix)

    **Example: Change measurement and outer kernel**

    .. jupyter-execute::

       import numpy as np
       from squlearn.encoding_circuit import ChebyshevTower
       from squlearn.kernel.lowlevel_kernel import ProjectedQuantumKernel
       from squlearn.util import Executor
       from squlearn.observables import CustomObservable
       from squlearn.kernel import QKRR

       fm = ChebyshevTower(num_qubits=4, num_features=1, num_chebyshev=4)

       # Create custom observables
       measuments = []
       measuments.append(CustomObservable(4,"ZZZZ"))
       measuments.append(CustomObservable(4,"YYYY"))
       measuments.append(CustomObservable(4,"XXXX"))

       # Use Matern Outer kernel with nu=0.5 as a outer kernel hyperparameter
       kernel = ProjectedQuantumKernel(encoding_circuit=fm,
                                       executor=Executor(),
                                       measurement=measuments,
                                       outer_kernel="matern",
                                       nu=0.5)
       ml_method = QKRR(quantum_kernel=kernel)

    Methods:
    --------
    """

    def __init__(
        self,
        encoding_circuit: EncodingCircuitBase,
        executor: Executor,
        measurement: Union[str, ObservableBase, list] = "XYZ",
        outer_kernel: Union[str, OuterKernelBase] = "gaussian",
        initial_parameters: Union[np.ndarray, None] = None,
        parameter_seed: Union[int, None] = 0,
        regularization: Union[str, None] = None,
        caching: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            encoding_circuit, executor, initial_parameters, parameter_seed, regularization
        )

        self._measurement_input = measurement
        self._outer_kernel_input = outer_kernel
        self._caching = caching
        self._derivative_cache = {}

        # Set-up measurement operator
        if isinstance(measurement, str):
            self._measurement = []
            for m_str in measurement:
                if m_str not in ("X", "Y", "Z"):
                    raise ValueError("Unknown measurement operator: {}".format(m_str))
                for i in range(self.num_qubits):
                    self._measurement.append(SinglePauli(self.num_qubits, i, op_str=m_str))
        elif isinstance(measurement, ObservableBase) or isinstance(measurement, list):
            self._measurement = measurement
        else:
            raise ValueError("Unknown type of measurement: {}".format(type(measurement)))

        # Set-up of the QNN
        self._qnn = LowLevelQNN(
            self._encoding_circuit, self._measurement, executor, caching=self._caching
        )

        # Set-up of the outer kernel
        self._set_outer_kernel(self._outer_kernel_input, **kwargs)

        # Generate default parameters of the measurement operators
        if initial_parameters is None:
            if self._parameters is None:
                self._parameters = np.array([])
            if isinstance(self._measurement, list):
                for i, m in enumerate(self._measurement):
                    self._parameters = np.concatenate(
                        (
                            self._parameters,
                            m.generate_initial_parameters(seed=parameter_seed + i + 1),
                        )
                    )
            elif isinstance(self._measurement, ObservableBase):
                self._parameters = np.concatenate(
                    (
                        self._parameters,
                        self._measurement.generate_initial_parameters(seed=parameter_seed),
                    )
                )
            else:
                raise ValueError("Unknown type of measurement: {}".format(type(measurement)))

        # Check if the number of parameters is correct
        if self._parameters is not None:
            if len(self._parameters) != self.num_parameters:
                raise ValueError(
                    "Number of initial parameters is wrong, expected number: {}".format(
                        self.num_parameters
                    )
                )

    def __reduce__(self):
        return (
            self.__class__,
            (
                self._encoding_circuit,
                self._executor,
                self._measurement_input,
                self._outer_kernel_input,
                self._parameters,
                self._parameter_seed,
                self._regularization,
                self._caching,
            ),
        )

    @property
    def num_features(self) -> int:
        """Feature dimension of the encoding circuit"""
        return self._qnn.num_features

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters of the encoding circuit"""
        return self._qnn.num_parameters + self._qnn.num_parameters_observable

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
        return self._qnn.evaluate(x, param, param_op, "f")["f"]

    def evaluate(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Evaluates the Projected Quantum Kernel for the given data points x and y.

        Args:
            x (np.ndarray): Data points x
            y (np.ndarray): Data points y, if None y = x is used
        Returns:
            The evaluated projected quantum kernel as numpy array
        """
        if self._parameters is None and self.num_parameters == 0:
            self._parameters = np.array([])

        if self._parameters is None:
            raise ValueError("Parameters have not been set yet!")

        kernel_matrix = self._outer_kernel(self._qnn, self._parameters, x, y)
        if (self._regularization is not None) and (
            kernel_matrix.shape[0] == kernel_matrix.shape[1]
        ):
            kernel_matrix = self._regularize_matrix(kernel_matrix)
        return kernel_matrix

    def evaluate_derivatives(
        self, x: np.ndarray, y: np.ndarray = None, values: Union[str, tuple] = "dKdx"
    ) -> dict:
        """
        Evaluates the Projected Quantum Kernel and its derivatives for the given data points x and y.

        Args:
            x (np.ndarray): Data points x
            y (np.ndarray): Data points y, if None y = x is used
            values (Union[str, tuple]): Values to evaluate. Can be a string or a tuple of strings.
            Possible values are: ``dKdx``, ``dKdy``, ``dKdxdx``, ``dKdp``
        Returns:
            Dictionary with the evaluated values

        """
        if self._parameters is None and self.num_parameters == 0:
            self._parameters = []
        if self._parameters is None:
            raise ValueError("Parameters have not been set yet!")
        param = self._parameters[: self._qnn.num_parameters]
        param_op = self._parameters[self._qnn.num_parameters :]

        if self._caching:
            caching_tuple = (
                to_tuple(x),
                to_tuple(param),
                to_tuple(param_op),
                (self._executor.shots == None),
            )
            value_dict = self._derivative_cache.get(caching_tuple, {})
        else:
            value_dict = {}

        value_dict["x"] = x
        value_dict["param"] = param
        value_dict["param_op"] = param_op

        def eval_helper(x, todo):
            return self._qnn.evaluate(x, param, param_op, todo)[todo]

        mutiple_values = True
        if isinstance(values, str):
            mutiple_values = False
            values = [values]

        for todo in values:
            if todo in value_dict:
                continue
            else:
                if todo == "K":
                    kernel_matrix = self.evaluate(x, y)
                elif todo == "dKdx" or todo == "dKdy":
                    if todo[2:] == "dx":
                        dOdx = eval_helper(x, "dfdx")
                    elif todo[2:] == "dy":
                        dOdx = eval_helper(y, "dfdx")

                    if self.num_features == 1:
                        kernel_matrix = np.einsum(
                            "njl,nl->nj",
                            self._outer_kernel.dKdx(
                                self._qnn, self._parameters, x, y, with_respect_to=todo[2:]
                            ),
                            dOdx[:, :, 0],
                        )  # shape (len(x), len(y))
                    else:
                        kernel_matrix = np.einsum(
                            "njl,nlm->mnj",
                            self._outer_kernel.dKdx(
                                self._qnn, self._parameters, x, y, with_respect_to=todo[2:]
                            ),
                            dOdx[:, :, :],
                        )  # shape (num_features, len(x), len(y))
                elif todo == "dKdp":
                    dOxdp = eval_helper(x, "dfdp")
                    dOydp = eval_helper(y, "dfdp")
                    kernel_matrix = np.einsum(
                        "njl,nlm->mnj",
                        self._outer_kernel.dKdx(
                            self._qnn, self._parameters, x, y, with_respect_to="dx"
                        ),
                        dOxdp[:, :, :],
                    ) + np.einsum(
                        "njl,nlm->mnj",
                        self._outer_kernel.dKdx(
                            self._qnn, self._parameters, x, y, with_respect_to="dy"
                        ),
                        dOydp[:, :, :],
                    )  # shape (num_parameters, len(x), len(y))
                elif todo == "dKdxdx":

                    if self.num_features > 1:
                        raise NotImplementedError(
                            "Second-order derivatives wrt multiple feature are not implemented"
                        )

                    dOdx = eval_helper(x, "dfdx")
                    dOdxdx = eval_helper(x, "dfdxdx")

                    first_term = np.einsum(
                        "njl,nl,nl->nj",
                        self._outer_kernel.dKdxdx(self._qnn, self._parameters, x, y),
                        dOdx[:, :, 0],
                        dOdx[:, :, 0],
                    )  # shape (len(x), len(y))
                    second_term = np.einsum(
                        "njl,nl->nj",
                        self._outer_kernel.dKdx(self._qnn, self._parameters, x, y),
                        dOdxdx[:, :, 0, 0],
                    )  # shape (len(x), len(y))
                    mixed_term = np.zeros((len(x), len(y)))  # i, j
                    for l in range(dOdx.shape[1]):
                        for m in range(dOdx.shape[1]):
                            if l != m:
                                mixed_term += 1 * np.einsum(
                                    "ij,i,i->ij",
                                    self._outer_kernel.dKdxdy(self._qnn, self._parameters, x, y)[
                                        :, :, l, m
                                    ],
                                    dOdx[:, l, 0],
                                    dOdx[:, m, 0],
                                )  # shape (len(x), len(y))
                    kernel_matrix = first_term + second_term + mixed_term
                else:
                    raise ValueError(f"{todo} is not implemented for single-dimensional data yet")

                value_dict[todo] = kernel_matrix

        if self._caching:
            self._derivative_cache[caching_tuple] = value_dict

        if mutiple_values:
            return value_dict
        else:
            return value_dict[values[0]]

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the Projected Quantum Kernel.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params(deep=False)
        params.update(self._outer_kernel.get_params())
        params["measurement"] = self._measurement_input
        params["num_qubits"] = self.num_qubits
        params["regularization"] = self._regularization
        params["outer_kernel"] = self._outer_kernel_input

        if deep:
            params.update(self._qnn.get_params())
        return params

    def set_params(self, **params):
        """
        Sets value of the Projected Quantum Kernel hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. ``num_qubits=2``
        """

        num_parameters_backup = self.num_parameters
        parameters_backup = self._parameters
        outer_kernel_input_backup = self._outer_kernel_input

        valid_params = self.get_params()
        for key in params.keys():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

        if "num_qubits" in params:
            self._encoding_circuit.set_params(num_qubits=params["num_qubits"])
            if isinstance(self._measurement_input, list):
                for m in self._measurement_input:
                    m.set_params(num_qubits=params["num_qubits"])
            elif isinstance(self._measurement_input, ObservableBase):
                self._measurement_input.set_params(num_qubits=params["num_qubits"])
            self.__init__(
                self._encoding_circuit,
                self._executor,
                self._measurement_input,
                self._outer_kernel,
                None,
                self._parameter_seed,
                self._regularization,
                self._caching,
            )
            params.pop("num_qubits")

        if "measurement" in params:
            self._measurement_input = params["measurement"]
            self.__init__(
                self._encoding_circuit,
                self._executor,
                self._measurement_input,
                self._outer_kernel,
                None,
                self._parameter_seed,
                self._regularization,
                self._caching,
            )
            params.pop("measurement")

        if "encoding_circuit" in params:
            self._encoding_circuit = params["encoding_circuit"]
            self.__init__(
                self._encoding_circuit,
                self._executor,
                self._measurement_input,
                self._outer_kernel,
                None,
                self._parameter_seed,
                self._regularization,
                self._caching,
            )
            params.pop("encoding_circuit")

        # Set parameters of the encoding circuit
        dict_ec = {}
        for key, value in params.items():
            if key in self._encoding_circuit.get_params():
                dict_ec[key] = value
        for key in dict_ec.keys():
            params.pop(key)
        if len(dict_ec) > 0:
            self._encoding_circuit.set_params(**dict_ec)
            self.__init__(
                self._encoding_circuit,
                self._executor,
                self._measurement_input,
                self._outer_kernel,
                None,
                self._parameter_seed,
                self._regularization,
                self._caching,
            )

        # Set Remaining QNN parameters
        dict_qnn = {}
        for key, value in params.items():
            if key in self._qnn.get_params():
                dict_qnn[key] = value
        for key in dict_qnn.keys():
            params.pop(key)
        if len(dict_qnn) > 0:
            self._qnn.set_params(**dict_qnn)

        # Set outer kernel
        if "outer_kernel" in params:
            self._outer_kernel_input = params["outer_kernel"]
            self._set_outer_kernel(self._outer_kernel_input)
            params.pop("outer_kernel")
        else:
            self._outer_kernel_input = outer_kernel_input_backup

        # Set outer kernel parameters
        dict_outer_kernel = {}
        valid_keys_outer_kernel = self._outer_kernel.get_params().keys()
        for key in params.keys():
            if key in valid_keys_outer_kernel:
                dict_outer_kernel[key] = value
        for key in dict_outer_kernel.keys():
            params.pop(key)
        if len(dict_outer_kernel) > 0:
            self._outer_kernel.set_params(**dict_outer_kernel)

        if "regularization" in params.keys():
            self._regularization = params["regularization"]
            params.pop("regularization")

        if self.num_parameters == num_parameters_backup:
            self._parameters = parameters_backup

        if len(params) > 0:
            raise ValueError("The following parameters could not be assigned:", params)

    @property
    def num_hyper_parameters(self) -> int:
        """The number of hyper-parameters of the outer kernel"""
        return self._outer_kernel.num_hyper_parameters

    @property
    def name_hyper_parameters(self) -> List[str]:
        """The names of the hyper-parameters of the outer kernel"""
        return self._outer_kernel.name_hyper_parameters

    def _set_outer_kernel(self, outer_kernel: Union[str, OuterKernelBase], **kwargs):
        """Private function for set-up the outer kernel

        Input can be a string for the sklearn outer kernels

        Args:
            outer_kernel (Union[str, OuterKernelBase]): OuterKernel that is applied to the
                                                        expectation values
            **kwargs: Keyword arguments for the outer kernel
        """
        if isinstance(outer_kernel, str):
            kwargs.pop("num_qubits", None)
            if outer_kernel.lower() == "gaussian":
                self._outer_kernel = GaussianOuterKernel(**kwargs)
            else:
                self._outer_kernel = OuterKernelBase.from_sklearn_kernel(outer_kernel, **kwargs)
        elif isinstance(outer_kernel, OuterKernelBase):
            self._outer_kernel = outer_kernel
        else:
            raise ValueError("Unknown type of outer kernel: {}".format(type(outer_kernel)))


class GaussianOuterKernel(OuterKernelBase):
    r"""
    Implementation of the Gaussian outer kernel:

    .. math::
        k(x_i, x_j) = \text{exp}\left(-\\gamma |(QNN(x_i)- QNN(x_j)|^2 \right)

    Args:
        gamma (float): hyperparameter :math:`\\gamma` of the Gaussian kernel
    """

    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self._num_hyper_parameters = 1
        self._name_hyper_parameters = ["gamma"]

    def __call__(
        self, qnn: LowLevelQNNBase, parameters: np.ndarray, x: np.ndarray, y: np.ndarray = None
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

        if len(param.shape) == 1 and len(param) == 1:
            param = float(param)
        if len(param_op.shape) == 1 and len(param_op) == 1:
            param_op = float(param_op)

        x_result = qnn.evaluate(x, param, param_op, "f")["f"]
        if y is not None:
            y_result = qnn.evaluate(y, param, param_op, "f")["f"]
        else:
            y_result = None

        return RBF(length_scale=1.0 / np.sqrt(2.0 * self.gamma))(x_result, y_result)

    def dKdx(
        self,
        qnn: LowLevelQNNBase,
        parameters: np.ndarray,
        x: np.ndarray,
        y: np.ndarray = None,
        with_respect_to: str = "dx",
    ) -> np.ndarray:
        """
        Implements the analytical derivative of the Gaussian kernel with respect to x.

        Args:
            qnn (QNN): QNN to be evaluated
            parameters (np.ndarray): parameters of the QNN
            x (np.ndarray): input data (n, num_features)
            y (np.ndarray): second optional input data (n, num_features)

        Returns:
            np.ndarray: derivative of the Gaussian projected kernel of shape (len(X), len(Y), num_qubits*len(measurement))
        """

        param = parameters[: qnn.num_parameters]
        param_op = parameters[qnn.num_parameters :]

        x_result = qnn.evaluate(x, param, param_op, "f")[
            "f"
        ]  # (n, num_qubits*len(measurement)*num_features) (i, l)
        if y is not None:
            y_result = qnn.evaluate(y, param, param_op, "f")[
                "f"
            ]  # (n, num_qubits*len(measurement)*num_features) (j, l)
        else:
            y_result = x_result

        coefficient_sign = -1 if with_respect_to == "dx" else 1
        return (
            coefficient_sign
            * 2
            * self.gamma
            * np.einsum(
                "ijl, ij -> ijl",
                (
                    x_result[:, None, :] - y_result
                ),  # difference of elements (i, l) and (j, l) [i, j, l]
                RBF(1.0 / np.sqrt(2.0 * self.gamma))(x_result, y_result),
            )
        )

    def dKdxdx(
        self, qnn: LowLevelQNNBase, parameters: np.ndarray, x: np.ndarray, y: np.ndarray = None
    ) -> np.ndarray:
        """
        Implements the analytical derivative of the Gaussian kernel with respect to x and x.

        Args:
            qnn (QNN): QNN to be evaluated
            parameters (np.ndarray): parameters of the QNN
            x (np.ndarray): input data
            y (np.ndarray): second optional input data

        Returns:
            np.ndarray: derivative dKdxdx of the Gaussian projected kernel shape (len(X), len(Y), num_qubits*len(measurement))
        """

        param = parameters[: qnn.num_parameters]
        param_op = parameters[qnn.num_parameters :]
        x_result = qnn.evaluate(x, param, param_op, "f")["f"]
        if y is not None:
            y_result = qnn.evaluate(y, param, param_op, "f")["f"]
        else:
            y_result = x_result

        return (2.0 * self.gamma) * np.einsum(
            "ijl, ij -> ijl",
            (
                2.0 * self.gamma * (x_result[:, None, :] - y_result) ** 2 - 1
            ),  # difference of elements (i, l) and (j, l) [i, j, l]
            RBF(1.0 / np.sqrt(2.0 * self.gamma))(x_result, y_result),
        )  # RBF kernel [i, j])

    def dKdxdy(
        self, qnn: LowLevelQNNBase, parameters: np.ndarray, x: np.ndarray, y: np.ndarray = None
    ) -> np.ndarray:
        """
        Implements the analytical derivative of the Gaussian kernel with respect to x and y.

        Args:
            qnn (QNN): QNN to be evaluated
            parameters (np.ndarray): parameters of the QNN
            x (np.ndarray): input data
            y (np.ndarray): second optional input data

        Returns:
            np.ndarray: derivative dKdxdy of the Gaussian projected kernel shape (len(X), len(Y), num_qubits*len(measurement), 1)
        """

        param = parameters[: qnn.num_parameters]
        param_op = parameters[qnn.num_parameters :]
        x_result = qnn.evaluate(x, param, param_op, "f")["f"]
        if y is not None:
            y_result = qnn.evaluate(y, param, param_op, "f")["f"]
        else:
            y_result = x_result

        return (
            4.0
            * self.gamma**2.0
            * np.einsum(
                "ijl,ij, ijp->ijlp",
                x_result[:, None, :] - y_result,
                RBF(1.0 / np.sqrt(2.0 * self.gamma))(x_result, y_result),
                x_result[:, None, :] - y_result,
            )
        )

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the Gaussian outer kernel.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = {"gamma": self.gamma}

        return params

    def set_params(self, **params) -> None:
        """
        Sets value of the Gaussian outer kernel hyper-parameters.

        Args:
            params: Hyper-parameters and their values
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )
            try:
                setattr(self, key, value)
            except:
                setattr(self, "_" + key, value)

        return None
