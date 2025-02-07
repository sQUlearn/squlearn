"""Fidelity Quantum Kernel class"""

import numpy as np
from typing import Union

from .kernel_matrix_base import KernelMatrixBase

from ...encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ...util.executor import Executor
from ...util.data_preprocessing import to_tuple


class FidelityKernelExpectationValue(KernelMatrixBase):
    """
    Fidelity Quantum Kernel based on the expectation value of the quantum circuit.

    Args:
        encoding_circuit (EncodingCircuitBase): The encoding circuit.
        executor (Executor): The executor for the quantum circuit.
        evaluate_duplicates (str): The evaluation mode for duplicates. Options are:
            - "all": Evaluate all duplicates.
            - "off_diagonal": Evaluate only off-diagonal duplicates.
            - "none": Do not evaluate any duplicates.
    """

    def __init__(
        self,
        encoding_circuit: EncodingCircuitBase,
        executor: Executor,
        evaluate_duplicates: str = "off_diagonal",
        caching: bool = True,
    ) -> None:

        self._encoding_circuit = encoding_circuit
        self._executor = executor
        self._evaluate_duplicates = evaluate_duplicates
        self._parameters = None
        self._qnn = None
        self._derivative_cache = {}
        self._caching = caching


    def assign_training_parameters(self, parameters: np.ndarray) -> None:
        """Assigns trainable parameters to the encoding circuit.

        Args:
            parameters (np.ndarray): Array of trainable parameters.
        """
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
        kernel_matrix = self.evaluate_derivatives(x, y, values="K")
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
            >>> to_FQK_circuit_format(x)
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
        # _qnn that implements a circuit U(y)^\dagger U(x) |0> and measure against P0=|0><0|^\otimes n, such that tr(\rho(x), \rho(y)) is obtained.
        # we use squlearn's circuit compose and inverse
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
    
    # def _pennylane_evaluate_kernel(self, x, y):
    #     """Function to evaluate the kernel matrix using PennyLane based on fidelity test.

    #     Args:
    #         x (np.ndarray): Vector of data for which the kernel matrix is evaluated
    #         y (np.ndarray): Vector of data for which the kernel matrix is evaluated
    #                         (can be similar to x)

    #     Returns:
    #         np.ndarray: Quantum kernel matrix as 2D numpy array.
    #     """

    #     def not_needed(i: int, j: int, x_i: np.ndarray, y_j: np.ndarray, symmetric: bool) -> bool:
    #         """Verifies if the kernel entry is trivial (to be set to `1.0`) or not.

    #         Args:
    #             i: Row index kernel matrix entry.
    #             j: Column index kernel matrix matrix entry.
    #             x_i: A sample from the dataset corresponding to the row in the kernel matrix.
    #             y_j: A sample from the dataset corresponding to the column in the kernel matrix.
    #             symmetric: Boolean indicating whether it is a symmetric case or not.

    #         Returns:
    #             True if value is trivial, False otherwise.
    #         """
    #         # evaluate all combinations -> all are needed
    #         if self._evaluate_duplicates == "all":
    #             return False

    #         # only off-diagonal entries are needed
    #         if symmetric and i == j and self._evaluate_duplicates == "off_diagonal":
    #             return True

    #         # don't evaluate any duplicates
    #         if np.array_equal(x_i, y_j) and self._evaluate_duplicates == "none":
    #             return True

    #         # otherwise evaluate
    #         return False

    #     is_symmetric = np.array_equal(x, y)
    #     num_features = x.shape[1]
    #     x_list = np.zeros((0, num_features))
    #     y_list = np.zeros((0, num_features))

    #     if is_symmetric:
    #         indices = []
    #         for i, x_i in enumerate(x):
    #             for j, x_j in enumerate(x[i:]):
    #                 if not_needed(i, i + j, x_i, x_j, True):
    #                     continue
    #                 x_list = np.vstack((x_list, x_i))
    #                 y_list = np.vstack((y_list, x_j))
    #                 indices.append((i, i + j))
    #     else:
    #         indices = []
    #         for i, x_i in enumerate(x):
    #             for j, y_j in enumerate(y):
    #                 if not_needed(i, j, x_i, y_j, False):
    #                     continue
    #                 x_list = np.vstack((x_list, x_i))
    #                 y_list = np.vstack((y_list, y_j))
    #                 indices.append((i, j))

    #     if self._parameter_vector is not None:
    #         if self._parameters is None:
    #             raise ValueError(
    #                 "Parameters have to been set with assign_parameters or as initial parameters!"
    #             )
    #         arguments = [(self._parameters, x1, x2) for x1, x2 in zip(y_list, x_list)]
    #     else:
    #         arguments = [(x1, x2) for x1, x2 in zip(y_list, x_list)]

    #     circuits = [self._pennylane_circuit] * len(arguments)
    #     all_probs = self._executor.pennylane_execute_batched(circuits, arguments)
    #     kernel_entries = [prob[0] for prob in all_probs]  # Get the count of the zero state

    #     kernel_matrix = np.ones((x.shape[0], y.shape[0]))
    #     if is_symmetric:
    #         for i, (col, row) in enumerate(indices):
    #             kernel_matrix[col, row] = kernel_entries[i]
    #             kernel_matrix[row, col] = kernel_entries[i]
    #     else:
    #         for i, (col, row) in enumerate(indices):
    #             kernel_matrix[col, row] = kernel_entries[i]

    #     return kernel_matrix
