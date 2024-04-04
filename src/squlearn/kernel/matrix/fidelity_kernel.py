""" Fidelity Quantum Kernel class"""

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
from qiskit.compiler import transpile
from qiskit_algorithms.utils import algorithm_globals

from .kernel_matrix_base import KernelMatrixBase
from ...encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ...util.executor import Executor


from ...util.pennylane.pennylane_gates import qiskit_pennyland_gate_dict
from ...util.pennylane.pennylane_circuit import PennyLaneCircuit

from functools import lru_cache


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
    ) -> None:
        super().__init__(
            encoding_circuit, executor, initial_parameters, parameter_seed, regularization
        )

        self._quantum_kernel = None
        self._evaluate_duplicates = evaluate_duplicates
        self._mit_depol_noise = mit_depol_noise

        if self._executor.quantum_framework == "pennylane":

            if self._executor.is_statevector:

                # Mode 1 for statevector: calculate the statevector of the quantum circuit
                # and use it to calculate the fidelity as the overlap of the two states.

                x = ParameterVector("x", self.num_features)
                if self.num_parameters > 0:
                    self._parameter_vector = ParameterVector("p", self.num_parameters)
                else:
                    self._parameter_vector = None

                enc_circ = self._encoding_circuit.get_circuit(x, self._parameter_vector)
                circuit = transpile(
                    enc_circ, basis_gates=qiskit_pennyland_gate_dict.keys(), optimization_level=0
                )
                self._pennylane_circuit = PennyLaneCircuit(circuit, "state", self._executor)
                def pennylane_circuit_executor(*args,**kwargs):
                    return self._executor.pennylane_execute(self._pennylane_circuit,*args,**kwargs)
                self._pennylane_circuit_cached = lru_cache()(pennylane_circuit_executor)

            else:

                # Mode 2 for qasm: calculate the |0> probabilities of the quantum circuit U(x)U(x)'
                x1 = ParameterVector("x1", self.num_features)
                x2 = ParameterVector("x2", self.num_features)
                if self.num_parameters > 0:
                    self._parameter_vector = ParameterVector("p", self.num_parameters)
                else:
                    self._parameter_vector = None

                enc_circ1 = self._encoding_circuit.get_circuit(x1, self._parameter_vector)
                enc_circ2 = self._encoding_circuit.get_circuit(x2, self._parameter_vector)

                circuit = enc_circ1.compose(enc_circ2.inverse())
                circuit = transpile(
                    circuit, basis_gates=qiskit_pennyland_gate_dict.keys(), optimization_level=0
                )
                pennylane_circuit = PennyLaneCircuit(circuit, "probs", self._executor)
                def pennylane_circuit_executor(*args,**kwargs):
                    return self._executor.pennylane_execute(pennylane_circuit,*args,**kwargs)
                self._pennylane_circuit = pennylane_circuit_executor

        elif self._executor.quantum_framework == "qiskit":

            self._feature_vector = ParameterVector("x", self.num_features)
            if self.num_parameters > 0:
                self._parameter_vector = ParameterVector("p", self.num_parameters)
            else:
                self._parameter_vector = None

            self._enc_circ = self._encoding_circuit.get_circuit(
                self._feature_vector, self._parameter_vector
            )
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
                fidelity = ComputeUncompute(sampler=self._executor.get_sampler())
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
        """
        Evaluates the fidelity kernel matrix.

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
        kernel_matrix = np.ones((x.shape[0], y.shape[0]))
        if self._executor.quantum_framework == "pennylane":
            # PennyLane implementation is found below, _pennylane_evaluate_kernel_sv replicates
            # FidelityStatevectorKernel from Qiskit Machine Learning
            if self._executor.is_statevector:
                kernel_matrix = self._pennylane_evaluate_kernel_sv(x, y)
            else:
                kernel_matrix = self._pennylane_evaluate_kernel(x, y)

        elif self._executor.quantum_framework == "qiskit":

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

    # Mitigating depolarizing noise after http://arxiv.org/abs/2105.02276v1
    def _get_msplit_kernel(self, kernel: np.ndarray) -> np.ndarray:
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
        mmean_kernel_matrix = np.zeros((kernel.shape[0], kernel.shape[1]))
        survival_prob_mean = self._survival_probability_mean(kernel)
        mmean_kernel_matrix = (
            kernel - 2 ** (-1.0 * self._num_qubits) * (1 - survival_prob_mean**2)
        ) / survival_prob_mean**2
        return mmean_kernel_matrix

    def _survival_probability(self, kernel: np.ndarray) -> np.ndarray:
        kernel_diagonal = np.diag(kernel)
        surv_prob = np.sqrt(
            (kernel_diagonal - 2 ** (-1.0 * self._num_qubits))
            / (1 - 2 ** (-1.0 * self._num_qubits))
        )
        return surv_prob

    def _survival_probability_mean(self, kernel: np.ndarray) -> float:
        surv_prob = self._survival_probability(kernel)
        return np.mean(surv_prob)

    def _pennylane_evaluate_kernel(self, x, y):

        def is_trivial(i: int, j: int, x_i: np.ndarray, y_j: np.ndarray, symmetric: bool) -> bool:
            """
            Verifies if the kernel entry is trivial (to be set to `1.0`) or not.

            Args:
                i: row index of the entry in the kernel matrix.
                j: column index of the entry in the kernel matrix.
                x_i: a sample from the dataset that corresponds to the row in the kernel matrix.
                y_j: a sample from the dataset that corresponds to the column in the kernel matrix.
                symmetric: whether it is a symmetric case or not.

            Returns:
                `True` if the entry is trivial, `False` otherwise.
            """
            # if we evaluate all combinations, then it is non-trivial
            if self._evaluate_duplicates == "all":
                return False

            # if we are on the diagonal and we don't evaluate it, it is trivial
            if symmetric and i == j and self._evaluate_duplicates == "off_diagonal":
                return True

            # if don't evaluate any duplicates
            if np.array_equal(x_i, y_j) and self._evaluate_duplicates == "none":
                return True

            # otherwise evaluate
            return False

        is_symmetric = np.array_equal(x, y)
        num_features = x.shape[1]
        x_list = np.zeros((0, num_features))
        y_list = np.zeros((0, num_features))

        if is_symmetric:
            indices = []
            for i, x_i in enumerate(x):
                for j, x_j in enumerate(x[i:]):
                    if is_trivial(i, i + j, x_i, x_j, True):
                        continue
                    x_list = np.vstack((x_list, x_i))
                    y_list = np.vstack((y_list, x_j))
                    indices.append((i, i + j))
        else:
            indices = []
            for i, x_i in enumerate(x):
                for j, y_j in enumerate(y):
                    if is_trivial(i, j, x_i, y_j, False):
                        continue
                    x_list = np.vstack((x_list, x_i))
                    y_list = np.vstack((y_list, y_j))
                    indices.append((i, j))

        if self._parameter_vector is not None:
            if self._parameters is None:
                raise ValueError(
                    "Parameters have to been set with assign_parameters or as initial parameters!"
                )
            kernel_entries = [
                self._pennylane_circuit(self._parameters, rp, lp)[0]
                for rp, lp in zip(y_list, x_list)
            ]

        else:
            kernel_entries = [self._pennylane_circuit(rp, lp)[0] for rp, lp in zip(y_list, x_list)]

        kernel_matrix = np.ones((x.shape[0], y.shape[0]))
        if is_symmetric:
            for i, (col, row) in enumerate(indices):
                kernel_matrix[col, row] = kernel_entries[i]
                kernel_matrix[row, col] = kernel_entries[i]
        else:
            for i, (col, row) in enumerate(indices):
                kernel_matrix[col, row] = kernel_entries[i]

        return kernel_matrix

    def _pennylane_evaluate_kernel_sv(self, x, y):

        sv_shots = self._executor.shots
        self._executor.set_shots(None)

        def compute_overlap(x: np.ndarray, y: np.ndarray) -> float:
            return np.abs(np.conj(x) @ y) ** 2

        def draw_shots(fidelity: float) -> float:
            return algorithm_globals.random.binomial(n=sv_shots, p=fidelity) / sv_shots

        def compute_kernel_entry(x: np.ndarray, y: np.ndarray) -> float:
            fidelity = compute_overlap(x, y)
            if sv_shots is not None:
                fidelity = draw_shots(fidelity)
            return fidelity

        if self._parameter_vector is not None:
            if self._parameters is None:
                raise ValueError(
                    "Parameters have to been set with assign_parameters or as initial parameters!"
                )
            x_sv = np.array(
                [self._pennylane_circuit_cached(tuple(self._parameters), tuple(x_)) for x_ in x]
            )
            y_sv = np.array(
                [self._pennylane_circuit_cached(tuple(self._parameters), tuple(y_)) for y_ in y]
            )
        else:
            x_sv = [self._pennylane_circuit_cached(tuple(x_)) for x_ in x]
            y_sv = [self._pennylane_circuit_cached(tuple(y_)) for y_ in y]

        kernel_matrix = np.ones((x.shape[0], y.shape[0]))
        for i, x_ in enumerate(x_sv):
            for j, y_ in enumerate(y_sv):
                if np.array_equal(x_, y_):
                    continue
                kernel_matrix[i, j] = compute_kernel_entry(x_, y_)

        self._executor.set_shots(sv_shots)

        return kernel_matrix
