"""Fidelity Quantum Kernel class"""

from typing import Union
import numpy as np

from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel,
    FidelityStatevectorKernel,
    TrainableFidelityQuantumKernel,
    TrainableFidelityStatevectorKernel,
)
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.circuit import ParameterVector

from .kernel_matrix_base import KernelMatrixBase
from ...encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ...util.executor import Executor, BaseSamplerV2
from ...util.data_preprocessing import convert_to_float64
from ...util.data_preprocessing import to_tuple

from .fidelity_kernel_pennylane import FidelityKernelPennyLane


class FidelityKernel(KernelMatrixBase):
    """
    Fidelity Quantum Kernel.

    The Fidelity Quantum Kernel is a based on the overlap of the quantum states.
    These quantum states
    can be defined by a parameterized quantum circuit. The Fidelity Quantum Kernel is defined as:

    .. math::

        K(x,y) = |\\langle \\phi(x) | \\phi(y) \\rangle|^2

    This class wraps to the respective Quantum Kernel implementations from `Qiskit Machine Learning
    <https://qiskit.org/ecosystem/machine-learning/apidocs/qiskit_machine_learning.kernels.html>`_.
    Depending on the choice of the backend and the choice of trainable parameters, the appropriate
    Quantum Kernel implementation is chosen.

    Args:
        encoding_circuit (EncodingCircuitBase): PQC encoding circuit.
        executor (Executor): Executor object.
        evaluate_duplicates (str), default='off_diagonal':
            Option for evaluating duplicates ('all', 'off_diagonal', 'none').
        mit_depol_noise (Union[str, None]), default=None:
            Option for mitigating depolarizing noise (``"msplit"`` or ``"mmean"``) after
            Ref. [4]. Only meaningful for
            FQKs computed on a real backend.
        initial_parameters (Union[np.ndarray, None], default=None):
            Initial parameters for the encoding circuit.
        parameter_seed (Union[int, None], default=0):
            Seed for the random number generator for the parameter initialization, if
            initial_parameters is None.
        regularization  (Union[str, None], default=None):
            Option for choosing different regularization techniques (``"thresholding"`` or
            ``"tikhonov"``) after Ref. [4] for the training kernel matrix, prior to  solving the
            linear system in the ``fit()``-procedure.

    References:
        [1]: `Havlicek et al., Supervised learning with quantum-enhanced feature spaces,
        Nature 567, 209-212 (2019).
        <https://www.nature.com/articles/s41586-019-0980-2>`_

        [2]: `Schuld et al., Quantum Machine Learning in Feature Hilbert Spaces,
        Phys. Rev. Lett. 122, 040504 (2019).
        <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.040504>`_

        [3]: `Schuld et al., Quantum Machine Learning Models are Kernel Methods:
        Noise-Enhanced Quantum Embeddings, arXiv:2105.02276 (2021).
        <https://arxiv.org/abs/2105.02276>`_

        [4]: `T. Hubregtsen et al.,
        "Training Quantum Embedding Kernels on Near-Term Quantum Computers",
        arXiv:2105.02276v1 (2021)
        <https://arxiv.org/abs/2105.02276>`_


    """

    def __init__(
        self,
        encoding_circuit: EncodingCircuitBase,
        executor: Executor,
        evaluate_duplicates: str = "off_diagonal",
        mit_depol_noise: Union[str, None] = None,
        initial_parameters: Union[np.ndarray, None] = None,
        parameter_seed: Union[int, None] = 0,
        regularization: Union[str, None] = None,
        caching: bool = False,
    ) -> None:
        super().__init__(
            encoding_circuit, executor, initial_parameters, parameter_seed, regularization
        )

        self._quantum_kernel = None
        self._evaluate_duplicates = evaluate_duplicates
        self._mit_depol_noise = mit_depol_noise
        self._qnn = None
        self._caching = caching
        self._derivative_cache = {}

        if self.num_parameters > 0:
            self._parameter_vector = ParameterVector("p", self.num_parameters)
        else:
            self._parameter_vector = None

        if self._executor.quantum_framework == "pennylane":
            self._quantum_kernel = FidelityKernelPennyLane(
                encoding_circuit=self._encoding_circuit,
                executor=self._executor,
                evaluate_duplicates=self._evaluate_duplicates,
            )

        elif self._executor.quantum_framework == "qiskit":
            # Underscore necessary to avoid name conflicts with the Qiskit quantum kernel
            self._feature_vector = ParameterVector("x_", self.num_features)

            self._enc_circ = self._encoding_circuit.get_circuit(
                self._feature_vector, self._parameter_vector
            )

            # Automatic select backend if not chosen
            if not self._executor.backend_chosen:
                self._enc_circ, _ = self._executor.select_backend(self._enc_circ)

            if self._executor.is_statevector:
                if self._parameter_vector is None:
                    self._quantum_kernel = FidelityStatevectorKernel(
                        feature_map=self._enc_circ,
                        shots=self._executor.get_shots(),
                        enforce_psd=False,
                    )
                else:
                    self._quantum_kernel = TrainableFidelityStatevectorKernel(
                        feature_map=self._enc_circ,
                        training_parameters=self._parameter_vector,
                        shots=self._executor.get_shots(),
                        enforce_psd=False,
                    )
            else:
                sampler = self._executor.get_sampler()
                if isinstance(sampler, BaseSamplerV2):
                    raise ValueError(
                        "Incompatible Qiskit version for Fidelity-Kernel calculation with Qiskit "
                        "Algorithms. Please downgrade to Qiskit 1.0 or consider using PennyLane."
                    )
                fidelity = ComputeUncompute(sampler=sampler)
                if self._parameter_vector is None:
                    self._quantum_kernel = FidelityQuantumKernel(
                        feature_map=self._enc_circ,
                        fidelity=fidelity,
                        evaluate_duplicates=self._evaluate_duplicates,
                        enforce_psd=False,
                    )
                else:
                    self._quantum_kernel = TrainableFidelityQuantumKernel(
                        feature_map=self._enc_circ,
                        fidelity=fidelity,
                        training_parameters=self._parameter_vector,
                        evaluate_duplicates=self._evaluate_duplicates,
                        enforce_psd=False,
                    )
        else:
            raise RuntimeError("Invalid quantum framework!")

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the fidelity kernel.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params(deep=False)
        params["evaluate_duplicates"] = self._evaluate_duplicates
        params["mit_depol_noise"] = self._mit_depol_noise
        params["regularization"] = self._regularization
        params["encoding_circuit"] = self._encoding_circuit
        if deep:
            params.update(self._encoding_circuit.get_params())
        return params

    def set_params(self, **params):
        """
        Sets value of the fidelity kernel hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. ``num_qubits=2``
        """
        num_parameters_backup = self.num_parameters
        parameters_backup = self._parameters

        # Check if all parameters are valid
        valid_params = self.get_params()
        for key in params.keys():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

        if "encoding_circuit" in params:
            self._encoding_circuit = params["encoding_circuit"]
            params.pop("encoding_circuit")

        dict_encoding_circuit = {}
        for key in params.keys():
            if key in self._encoding_circuit.get_params().keys():
                dict_encoding_circuit[key] = params[key]
        for key in dict_encoding_circuit.keys():
            params.pop(key)

        self._encoding_circuit.set_params(**dict_encoding_circuit)

        if "evaluate_duplicates" in params.keys():
            self._evaluate_duplicates = params["evaluate_duplicates"].lower()
            params.pop("evaluate_duplicates")
        if "mit_depol_noise" in params.keys():
            self._mit_depol_noise = params["mit_depol_noise"]
            params.pop("mit_depol_noise")
        if "regularization" in params.keys():
            self._regularization = params["regularization"]
            params.pop("regularization")

        self.__init__(
            self._encoding_circuit,
            self._executor,
            self._evaluate_duplicates,
            self._mit_depol_noise,
            None,
            self._parameter_seed,
            self._regularization,
        )

        if self.num_parameters == num_parameters_backup:
            self._parameters = parameters_backup

        if len(params) > 0:
            raise ValueError("The following parameters could not be assigned:", params)

    def evaluate(self, x: np.ndarray, y: Union[np.ndarray, None] = None) -> np.ndarray:
        """Evaluates the fidelity kernel matrix.

        Args:
            x (np.ndarray) :
                Vector of training or test data for which the kernel matrix is evaluated
            y (np.ndarray, default=None) :
                Vector of training or test data for which the kernel matrix is evaluated
        Returns:
            Returns the quantum kernel matrix as 2D numpy array.
        """

        if y is None:
            y = x

        x = convert_to_float64(x)
        y = convert_to_float64(y)

        if self._parameter_vector is not None:
            if self._parameters is None:
                raise ValueError(
                    "Parameters have to been set with assign_parameters or as initial parameters!"
                )
            self._quantum_kernel.assign_training_parameters(self._parameters)

        kernel_matrix = self._quantum_kernel.evaluate(x, y)

        if self._mit_depol_noise is not None:
            print("WARNING: Advanced option. Do not use it within an squlearn.kernel.ml workflow")
            if not np.array_equal(x, y):
                raise ValueError(
                    "Mitigating depolarizing noise works only for square matrices computed on real"
                    " backend"
                )
            else:
                if self._mit_depol_noise == "msplit":
                    kernel_matrix = self._get_msplit_kernel(kernel_matrix)
                elif self._mit_depol_noise == "mmean":
                    kernel_matrix = self._get_mmean_kernel(kernel_matrix)

        if (self._regularization is not None) and (
            kernel_matrix.shape[0] == kernel_matrix.shape[1]
        ):
            kernel_matrix = self._regularize_matrix(kernel_matrix)
        return kernel_matrix

    def evaluate_derivatives(
        self, x: np.ndarray, y: np.ndarray = None, values: Union[str, tuple] = "dKdx"
    ) -> dict:
        """
        Evaluates the Fidelity Quantum Kernel and its derivatives for the given data points x and y.

        Args:
            x (np.ndarray): Data points x
            y (np.ndarray): Data points y, if None y = x is used
            values (Union[str, tuple]): Values to evaluate. Can be a string or a tuple of strings.
                Possible values are: ``dKdx``, ``dKdy``, ``dKdxdx``, ``dKdydy``, ``dKdxdy``, ``dKdydx``, ``dKdp`` and ``jacobian``.
        Returns:
            Dictionary with the evaluated values

        """
        from squlearn.qnn.lowlevel_qnn import LowLevelQNN

        def P0_squlearn(num_qubits):
            """
            Create the P0 observable: (|0><0|)^\otimes n for the quantum circuit in the format of the squlearn library.
            Note that |0><0| = 0.5*(I + Z)

            Parameters:
            num_qubits: int, the number of qubits in the quantum circuit.

            return:
            - CustomObservable: The P0 observable in the format of the squlearn library.
            """
            from qiskit.quantum_info import SparsePauliOp
            from squlearn.observables import CustomObservable

            P0_single_qubit = SparsePauliOp.from_list([("Z", 0.5), ("I", 0.5)])
            P0_temp = P0_single_qubit
            for i in range(1, num_qubits):
                P0_temp = P0_temp.expand(P0_single_qubit)
            observable_tuple_list = P0_temp.to_list()
            pauli_str = [observable[0] for observable in observable_tuple_list]
            return CustomObservable(num_qubits, pauli_str, parameterized=True)

        def to_FQK_circuit_format(x, y=None):
            """
            Transforms an input array of shape (n, m) into an array of shape (n*n, 2*m),
            where each row consists of all possible ordered pairs of rows from the input array.

            Parameters:
            x (numpy.ndarray): An input array of shape (n, m), where n is the number of samples
                            and m is the number of features.

            Returns:
            numpy.ndarray: A transformed array of shape (n*n, 2*m), containing all possible
                        ordered pairs of rows from x.

            Example:
            --------
            >>> x = np.array([[1],
            ...               [2],
            ...               [3]])
            >>> to_proper_format(x)
            array([[1, 1],
                [2, 1],
                [3, 1],
                [1, 2],
                [2, 2],
                [3, 2],
                [1, 3],
                [2, 3],
                [3, 3]])
            """
            if y is None:
                y = x
                n = x.shape[0]
                x_rep = np.repeat(x, n, axis=0)  # Repeat each row n times
                x_tile = np.tile(x, (n, 1))  # Tile the entire array n times
            else:
                n = x.shape[0]
                n2 = y.shape[0]
                x_rep = np.repeat(x, n2, axis=0)
                x_tile = np.tile(y, (n, 1))
            result = np.hstack((x_rep, x_tile))
            return result

        if self._parameters is None and self.num_parameters == 0:
            self._parameters = []
        if self._parameters is None:
            raise ValueError("Parameters have not been set yet!")
        coef = np.array(
            [
                1 / 2**self.encoding_circuit.num_qubits
                for i in range(2**self.encoding_circuit.num_qubits)
            ]
        )
        # _qnn that implements a circuit U(y)^\dagger U(x) |0> and measuring P0=|0><0|^\otimes n, such that tr(\rho(x), \rho(y)) is obtained.
        # we use squlearn circuits compose and inverse
        self._qnn = LowLevelQNN(
            (self.encoding_circuit).compose(self.encoding_circuit.inverse()),
            P0_squlearn(self.encoding_circuit.num_qubits),
            executor=self._executor,
        )

        param = self._parameters
        param_op = coef

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

        value_dict["x"] = to_FQK_circuit_format(
            x, y
        )  # from shape: (n1, m) and (n2, m) to shape: (n1*n2, 2*m)
        value_dict["param"] = param  # Parameters of Quantum Kernel
        value_dict["param_op"] = param_op  # Constant coefficients for the observable P0

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
                    kernel_matrix = eval_helper(value_dict["x"], "f").reshape(
                        x.shape[0],
                        y.shape[0],
                    )
                elif todo == "dKdx" or todo == "dKdy":
                    dKdx = eval_helper(value_dict["x"], "dfdx").reshape(
                        x.shape[0], y.shape[0], 2 * self.num_features
                    )  # shape (len(x), len(y), 2*num_features)
                    # to keep consistency with the PQK derivatives, we need to transpose the dKdx matrix to be of shape (2*num_features, len(x), len(y))
                    dKdx = dKdx.transpose(2, 0, 1)
                    if self.num_features == 1:
                        if todo[2:] == "dx":
                            kernel_matrix = dKdx[0]
                        elif todo[2:] == "dy":
                            kernel_matrix = dKdx[1]
                    else:
                        if todo[2:] == "dx":
                            kernel_matrix = dKdx[: self.num_features]
                        elif todo[2:] == "dy":
                            kernel_matrix = dKdx[self.num_features :]
                elif todo == "dKdp":
                    dKdp = eval_helper(value_dict["x"], "dfdp").reshape(
                        x.shape[0], y.shape[0], self.num_parameters
                    )  # shape (len(x), len(y), num_parameters)
                    # to keep consistency with the PQK derivatives, we need to transpose the dKdp matrix to be of shape (num_parameters, len(x), len(y))
                    kernel_matrix = dKdp.transpose(2, 0, 1)
                elif (
                    todo == "dKdxdx"
                    or todo == "dKdydy"
                    or todo == "dKdxdy"
                    or todo == "dKdydx"
                    or todo == "dKdxdy"
                    or todo == "jacobian"
                ):
                    jacobian = eval_helper(value_dict["x"], "dfdxdx").reshape(
                        x.shape[0], y.shape[0], 2 * self.num_features, 2 * self.num_features
                    )  # shape (len(x), len(y), 2*num_features, 2*num_features)
                    # to keep consistency with the PQK derivatives, we need to transpose the jacobian matrix to be of shape (2*num_features, 2*num_features, len(x), len(y))
                    jacobian = jacobian.transpose(2, 3, 0, 1)
                    if self.num_features == 1:
                        if todo[2:] == "dxdx":
                            kernel_matrix = jacobian[0, 0]  # shape (len(x), len(x))
                        elif todo[2:] == "dydy":
                            kernel_matrix = jacobian[1, 1]  # shape (len(y), len(y))
                        elif todo[2:] == "dxdy":
                            kernel_matrix = jacobian[0, 1]  # shape (len(x), len(y))
                        elif todo[2:] == "dydx":
                            kernel_matrix = jacobian[1, 0]  # shape (len(y), len(x))
                        elif todo == "jacobian":
                            kernel_matrix = jacobian
                    else:
                        if todo[2:] == "dxdx":
                            kernel_matrix = jacobian[
                                : self.num_features, : self.num_features
                            ]  # shape (num_features, num_features, len(x), len(x))
                        elif todo[2:] == "dydy":
                            kernel_matrix = jacobian[
                                self.num_features :, self.num_features :
                            ]  # shape (num_features, num_features, len(y), len(y))
                        elif todo[2:] == "dxdy":
                            kernel_matrix = jacobian[
                                : self.num_features, self.num_features :
                            ]  # shape (num_features, num_features, len(x), len(y))
                        elif todo[2:] == "dydx":
                            kernel_matrix = jacobian[
                                self.num_features :, : self.num_features
                            ]  # shape (num_features, num_features, len(y), len(x))
                        elif todo == "jacobian":
                            kernel_matrix = (
                                jacobian  # shape (2*num_features, 2*num_features, len(x), len(y))
                            )
                value_dict[todo] = kernel_matrix

        if self._caching:
            self._derivative_cache[caching_tuple] = value_dict

        if mutiple_values:
            return value_dict
        else:
            return value_dict[values[0]]

    def _get_msplit_kernel(self, kernel: np.ndarray) -> np.ndarray:
        """Function to mitigate depolarizing noise using msplit method.

        Mitigating depolarizing noise after http://arxiv.org/abs/2105.02276v1

        Args:
            kernel (np.ndarray): Quantum kernel matrix as 2D numpy array.

        Returns:
            np.ndarray: Mitigated Quantum kernel matrix as 2D numpy array.
        """

        msplit_kernel_matrix = np.zeros((kernel.shape[0], kernel.shape[1]))
        survival_prob = self._survival_probability(kernel)
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                msplit_kernel_matrix[i, j] = (
                    kernel[i, j]
                    - 2 ** (-1.0 * self._num_qubits) * (1 - survival_prob[i] * survival_prob[j])
                ) / (survival_prob[i] * survival_prob[j])
        return msplit_kernel_matrix

    def _get_mmean_kernel(self, kernel: np.ndarray) -> np.ndarray:
        """
        Function to mitigate depolarizing noise using mmean method.

        Args:
            kernel (np.ndarray): Quantum kernel matrix as 2D numpy array.

        Returns:
            np.ndarray: Mitigated Quantum kernel matrix as 2D numpy array.
        """
        mmean_kernel_matrix = np.zeros((kernel.shape[0], kernel.shape[1]))
        survival_prob_mean = self._survival_probability_mean(kernel)
        mmean_kernel_matrix = (
            kernel - 2 ** (-1.0 * self._num_qubits) * (1 - survival_prob_mean**2)
        ) / survival_prob_mean**2
        return mmean_kernel_matrix

    def _survival_probability(self, kernel: np.ndarray) -> np.ndarray:
        """Function to calculate the survival probability.

        Args:
            kernel (np.ndarray): Quantum kernel matrix as 2D numpy array.

        Returns:
            np.ndarray: Survival probability as 1D numpy array.
        """
        kernel_diagonal = np.diag(kernel)
        surv_prob = np.sqrt(
            (kernel_diagonal - 2 ** (-1.0 * self._num_qubits))
            / (1 - 2 ** (-1.0 * self._num_qubits))
        )
        return surv_prob

    def _survival_probability_mean(self, kernel: np.ndarray) -> float:
        """
        Function to calculate the mean survival probability.

        Args:
            kernel (np.ndarray): Quantum kernel matrix as 2D numpy array.

        Returns:
            float: Mean survival probability.
        """
        surv_prob = self._survival_probability(kernel)
        return np.mean(surv_prob)
