"""Fidelity Quantum Kernel class"""

from typing import Union
from functools import lru_cache
import numpy as np

from qiskit.circuit import ParameterVector
from qiskit.compiler import transpile
from qiskit_algorithms.utils import algorithm_globals

from ...encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ...util.executor import Executor

from ...util.pennylane.pennylane_gates import qiskit_pennylane_target
from ...util.pennylane.pennylane_circuit import PennyLaneCircuit

from ...util.qulacs.qulacs_circuit import QulacsCircuit
from ...util.qulacs.qulacs_execution import qulacs_evaluate_statevector

from ...util.data_preprocessing import to_tuple, adjust_features


class FidelityKernelStatevector:
    """
    Fidelity Quantum Kernel implementation based on PennyLane.

    Args:
        encoding_circuit (EncodingCircuitBase): The encoding circuit.
        executor (Executor): The executor for the quantum circuit.
        num_features (int): The number of features in the input data.
        evaluate_duplicates (str): The evaluation mode for duplicates. Options are:
            - "all": Evaluate all duplicates.
            - "off_diagonal": Evaluate only off-diagonal duplicates.
            - "none": Do not evaluate any duplicates.
        cache_size (int): The cache size for the lru_cache.
    """

    def __init__(
        self,
        encoding_circuit: EncodingCircuitBase,
        executor: Executor,
        num_features: int,
        evaluate_duplicates: str = "off_diagonal",
        cache_size=None,
    ) -> None:

        self._encoding_circuit = encoding_circuit
        self._executor = executor
        self._evaluate_duplicates = evaluate_duplicates
        self._cache_size = cache_size
        self._num_features = num_features
        self._parameters = None

        if self._executor.quantum_framework not in ["pennylane", "qulacs"]:
            raise NotImplementedError(
                "FidelityKernelStatevector is not supported for this quantum framework:",
                self._executor.quantum_framework,
            )

        if self._executor.is_statevector:

            # Mode 1 for statevector: calculate the statevector of the quantum circuit
            # and use it to calculate the fidelity as the overlap of the two states.

            x = ParameterVector("x", self._num_features)
            if self.num_parameters > 0:
                self._parameter_vector = ParameterVector("p", self.num_parameters)
            else:
                self._parameter_vector = None

            enc_circ = self._encoding_circuit.get_circuit(x, self._parameter_vector)

            if self._executor.quantum_framework == "pennylane":
                circuit = transpile(enc_circ, target=qiskit_pennylane_target, optimization_level=0)
                self._pennylane_circuit = PennyLaneCircuit(circuit, "state")

                @lru_cache(maxsize=self._cache_size)
                def pennylane_circuit_executor(*args, **kwargs):
                    args_numpy = [np.array(arg) for arg in args]
                    return self._executor.pennylane_execute(
                        self._pennylane_circuit, *args_numpy, **kwargs
                    )

                self._cached_execution = pennylane_circuit_executor

            elif self._executor.quantum_framework == "qulacs":

                enc_circ = self._encoding_circuit.get_circuit(x, self._parameter_vector)
                self._qulacs_circuit = QulacsCircuit(enc_circ, None)

                @lru_cache(maxsize=self._cache_size)
                def qulacs_circuit_executor(*args):
                    args_numpy = [np.array(arg) for arg in args]
                    if len(args_numpy) == 0:
                        return self._executor.qulacs_execute(
                            qulacs_evaluate_statevector, self._qulacs_circuit
                        )
                    elif len(args_numpy) == 1:
                        return self._executor.qulacs_execute(
                            qulacs_evaluate_statevector, self._qulacs_circuit, x=args_numpy[0]
                        )
                    elif len(args_numpy) == 2:
                        return self._executor.qulacs_execute(
                            qulacs_evaluate_statevector,
                            self._qulacs_circuit,
                            p=args_numpy[0],
                            x=args_numpy[1],
                        )

                self._cached_execution = qulacs_circuit_executor

            else:
                raise RuntimeError(
                    "Quantum framework not supported for FidelityKernelStatevector: "
                    f"{self._executor.quantum_framework}"
                )

        else:

            # Mode 2 for shot based: calculate the |0> probabilities
            # of the quantum circuit U(x)U(x)'

            if self._executor.quantum_framework == "pennylane":
                x1 = ParameterVector("x1", self._num_features)
                x2 = ParameterVector("x2", self._num_features)
                if self.num_parameters > 0:
                    self._parameter_vector = ParameterVector("p", self.num_parameters)
                else:
                    self._parameter_vector = None

                enc_circ1 = self._encoding_circuit.get_circuit(x1, self._parameter_vector)
                enc_circ2 = self._encoding_circuit.get_circuit(x2, self._parameter_vector)

                circuit = enc_circ1.compose(enc_circ2.inverse())
                circuit = transpile(circuit, target=qiskit_pennylane_target, optimization_level=0)
                self._pennylane_circuit = PennyLaneCircuit(circuit, "probs")
            elif self._executor.quantum_framework == "qulacs":
                raise NotImplementedError(
                    "Shot based fidelity kernel is not implemented for Qulacs yet."
                )
            else:
                raise RuntimeError(
                    "Quantum framework not supported for FidelityKernelStatevector: "
                    f"{self._executor.quantum_framework}"
                )

    @property
    def num_parameters(self) -> int:
        """Returns the number of trainable parameters."""
        return self._encoding_circuit.num_parameters

    def assign_training_parameters(self, parameters: np.ndarray) -> None:
        """Assigns trainable parameters to the encoding circuit.

        Args:
            parameters (np.ndarray): Array of trainable parameters.
        """
        if self._parameter_vector is None:
            raise ValueError("No trainable parameters are available for assignment.")
        if len(parameters) != self.num_parameters:
            raise ValueError(
                "Number of parameters does not match the number of trainable parameters."
            )
        self._parameters = parameters

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

        kernel_matrix = np.ones((x.shape[0], y.shape[0]))

        if self._executor.is_statevector:
            kernel_matrix = self.evaluate_kernel_sv(x, y)
        else:
            kernel_matrix = self.evaluate_kernel_shots(x, y)

        return kernel_matrix

    def evaluate_kernel_shots(self, x, y):
        """Function to evaluate the kernel matrix using PennyLane based on fidelity test.

        Args:
            x (np.ndarray): Vector of data for which the kernel matrix is evaluated
            y (np.ndarray): Vector of data for which the kernel matrix is evaluated
                            (can be similar to x)

        Returns:
            np.ndarray: Quantum kernel matrix as 2D numpy array.
        """

        if self._executor.quantum_framework != "pennylane":
            raise RuntimeError(
                "Quantum framework not supported for FidelityKernelStatevector: "
                f"{self._executor.quantum_framework}"
            )

        def not_needed(i: int, j: int, x_i: np.ndarray, y_j: np.ndarray, symmetric: bool) -> bool:
            """Verifies if the kernel entry is trivial (to be set to `1.0`) or not.

            Args:
                i: Row index kernel matrix entry.
                j: Column index kernel matrix matrix entry.
                x_i: A sample from the dataset corresponding to the row in the kernel matrix.
                y_j: A sample from the dataset corresponding to the column in the kernel matrix.
                symmetric: Boolean indicating whether it is a symmetric case or not.

            Returns:
                True if value is trivial, False otherwise.
            """
            # evaluate all combinations -> all are needed
            if self._evaluate_duplicates == "all":
                return False

            # only off-diagonal entries are needed
            if symmetric and i == j and self._evaluate_duplicates == "off_diagonal":
                return True

            # don't evaluate any duplicates
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
                    if not_needed(i, i + j, x_i, x_j, True):
                        continue
                    x_list = np.vstack((x_list, x_i))
                    y_list = np.vstack((y_list, x_j))
                    indices.append((i, i + j))
        else:
            indices = []
            for i, x_i in enumerate(x):
                for j, y_j in enumerate(y):
                    if not_needed(i, j, x_i, y_j, False):
                        continue
                    x_list = np.vstack((x_list, x_i))
                    y_list = np.vstack((y_list, y_j))
                    indices.append((i, j))

        if self._parameter_vector is not None:
            if self._parameters is None:
                raise ValueError(
                    "Parameters have to been set with assign_parameters or as initial parameters!"
                )
            arguments = [(self._parameters, x1, x2) for x1, x2 in zip(y_list, x_list)]
        else:
            arguments = [(x1, x2) for x1, x2 in zip(y_list, x_list)]

        circuits = [self._pennylane_circuit] * len(arguments)
        all_probs = self._executor.pennylane_execute_batched(circuits, arguments)
        kernel_entries = [prob[0] for prob in all_probs]  # Get the count of the zero state

        kernel_matrix = np.ones((x.shape[0], y.shape[0]))
        if is_symmetric:
            for i, (col, row) in enumerate(indices):
                kernel_matrix[col, row] = kernel_entries[i]
                kernel_matrix[row, col] = kernel_entries[i]
        else:
            for i, (col, row) in enumerate(indices):
                kernel_matrix[col, row] = kernel_entries[i]

        return kernel_matrix

    def evaluate_kernel_sv(self, x, y):
        """
        Function to evaluate the kernel matrix with statevector simulator using PennyLane.

        Evaluates the kernel matrix using the statevectors, overlap is then
        classically calculated.

        Args:
            x (np.ndarray): Vector of data for which the kernel matrix is evaluated
            y (np.ndarray): Vector of data for which the kernel matrix is evaluated
                            (can be similar to x)

        Returns:
            np.ndarray: Quantum kernel matrix as 2D numpy array.
        """
        shots = self._executor.shots
        self._executor.set_shots(None)
        is_symmetric = np.array_equal(x, y)

        def get_kernel_entry(x: np.ndarray, y: np.ndarray) -> float:
            """Compute the kernel entry based on the overlap x and y."""
            # Calculate overlap between statevector x and y
            overlap = np.abs(np.matmul(x.conj(), y)) ** 2
            # If shots are set, draw from the binomial distribution
            if shots is not None and shots > 1:
                overlap = algorithm_globals.random.binomial(n=shots, p=overlap) / shots
            return overlap

        # Convert the input data to the correct format for the lrucache
        x_inp, _ = adjust_features(x, self._num_features)
        x_inpT = to_tuple(np.transpose(x_inp), flatten=False)
        y_inp, _ = adjust_features(y, self._num_features)
        y_inpT = to_tuple(np.transpose(y_inp), flatten=False)

        if self._executor.quantum_framework == "pennylane":

            if self._parameter_vector is not None:
                if self._parameters is None:
                    raise ValueError(
                        "Parameters have to been set with assign_parameters or as initial parameters!"
                    )
                x_sv = np.array(self._cached_execution(tuple(self._parameters), x_inpT))
                y_sv = np.array(self._cached_execution(tuple(self._parameters), y_inpT))
            else:
                x_sv = np.array(self._cached_execution(x_inpT))
                y_sv = np.array(self._cached_execution(y_inpT))

        elif self._executor.quantum_framework == "qulacs":

            if self._parameter_vector is not None:
                if self._parameters is None:
                    raise ValueError(
                        "Parameters have to been set with assign_parameters or as initial parameters!"
                    )
                x_sv = np.array(
                    [self._cached_execution(tuple(self._parameters), tuple(x_)) for x_ in x_inp]
                )
                y_sv = np.array(
                    [self._cached_execution(tuple(self._parameters), tuple(y_)) for y_ in y_inp]
                )
            else:
                x_sv = np.array([self._cached_execution(tuple(x_)) for x_ in x_inp])
                y_sv = np.array([self._cached_execution(tuple(y_)) for y_ in y_inp])
        else:
            raise RuntimeError(
                "Quantum framework not supported for FidelityKernelStatevector: "
                f"{self._executor.quantum_framework}"
            )

        if len(x_sv.shape) == 1:
            x_sv = np.array([x_sv])
        if len(y_sv.shape) == 1:
            y_sv = np.array([y_sv])

        kernel_matrix = np.zeros((x.shape[0], y.shape[0]))

        if is_symmetric:
            # pylint: disable-next=consider-using-enumerate
            for i in range(len(x_sv)):
                # pylint: disable-next=consider-using-enumerate
                for j in range(i):
                    if np.array_equal(x_sv[i], y_sv[j]):
                        if self._evaluate_duplicates == "none":
                            kernel_matrix[i, j] = 1.0
                            kernel_matrix[j, i] = 1.0
                            continue
                    kernel_matrix[i, j] = get_kernel_entry(x_sv[i], y_sv[j])
                    kernel_matrix[j, i] = kernel_matrix[i, j]
        else:
            for i, x_ in enumerate(x_sv):
                for j, y_ in enumerate(y_sv):
                    if np.array_equal(x_, y_):
                        if self._evaluate_duplicates == "none":
                            kernel_matrix[i, j] = 1.0
                            continue
                    kernel_matrix[i, j] = get_kernel_entry(x_, y_)

        if self._evaluate_duplicates in ["none", "off_diagonal"] and is_symmetric:
            for i in range(x.shape[0]):
                kernel_matrix[i, i] = 1.0

        self._executor.set_shots(shots)

        return kernel_matrix
