"""Fidelity Quantum Kernel class"""

import numpy as np
from typing import Union

from qiskit.quantum_info import SparsePauliOp
from squlearn.observables import CustomObservable

from .kernel_matrix_base import KernelMatrixBase

from ...encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ...util.executor import Executor
from ...util.data_preprocessing import to_tuple

from ...qnn.lowlevel_qnn import LowLevelQNN


class FidelityKernelExpectationValue(KernelMatrixBase):
    r"""
    Fidelity Quantum Kernel evaluation based on quantum circuit and expectation values.

    Fidelity Quantum Kernel based on the expectation value of the quantum circuit constructed by
    evaluating the expectation value of the observable :math:`P_0 = |0\rangle\langle0|^{\otimes n}`
    with the quantum circuit :math:`U(y)^{\dagger} U(x) |0\rangle`.

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
        evaluate_duplicates: str = "none",
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
        Evaluates the Fidelity Kernel and its derivatives for the given data points x and y.

        Args:
            x (np.ndarray): Data points x
            y (np.ndarray): Data points y, if None y = x is used
            values (Union[str, tuple]): Values to evaluate. Can be a string or a tuple of strings.
                Possible values are:
                ``dKdx``, ``dKdy``, ``dKdxdx``, ``dKdydy``, ``dKdxdy``, ``dKdydx``, ``dKdp``
                and ``jacobian``.
        Returns:
            Dictionary with the evaluated values

        """

        def P0_operator(num_qubits):
            r"""
                Function for creating the |0><0| oberservable

                Creates the :math:`P_0` observable: :math:`(|0\rangle\l\angle0|)^{\otimes n}` for
                the quantum circuit in the format of the squlearn library.
                Note that :math:`|0\rangle\langle 0| = 0.5 \cdot (I + Z)`.

            Args:
                num_qubits (int): Number of qubits in the quantum circuit.

            return:
                CustomObservable: The P0 observable in the squlearn library format
            """
            P0_single_qubit = SparsePauliOp.from_list([("Z", 0.5), ("I", 0.5)])
            P0_temp = P0_single_qubit
            for _ in range(1, num_qubits):
                P0_temp = P0_temp.expand(P0_single_qubit)
            observable_tuple_list = P0_temp.simplify().simplify().to_list()
            pauli_str = [observable[0] for observable in observable_tuple_list]
            return CustomObservable(num_qubits, pauli_str, parameterized=True)

        def get_flattened_matrix_indices(n, part="lower"):
            """
            Returns the indices of the lower triangle or diagonal elements of a matrix in flattened
            form.

            Args:
                n (int): Size of the matrix (n x n).
                part (str): Part of the matrix to return indices for. Options are "lower" for
                            lower triangle and "diagonal" for diagonal elements.

            Returns:
                numpy.ndarray: Indices in flattened form.
            """
            matrix = np.arange(n**2).reshape(n, n)  # Creating a sample n x n matrix
            flat_matrix = matrix.ravel()  # Flattening the matrix

            if part == "lower":
                lower_tri_rows, lower_tri_cols = np.tril_indices(n, k=-1)
                lower_tri_flat_indices = np.ravel_multi_index(
                    (lower_tri_rows, lower_tri_cols), (n, n)
                )
                return lower_tri_flat_indices
            elif part == "higher":
                upper_tri_rows, upper_tri_cols = np.triu_indices(n, k=1)
                upper_tri_flat_indices = np.ravel_multi_index(
                    (upper_tri_rows, upper_tri_cols), (n, n)
                )
                return upper_tri_flat_indices
            elif part == "diagonal" or part == "off_diagonal":
                diag_indices = np.arange(0, n * (n + 1), n + 1)
                remaining_indices = np.setdiff1d(flat_matrix, diag_indices)
                if part == "diagonal":
                    return diag_indices
                return remaining_indices
            else:
                raise ValueError("Invalid part specified. Use 'lower' or 'diagonal'.")

        def to_FQK_circuit_format(x, y=None, evaluate_duplicates="all"):
            """
            Transforms an input array of shape (n, m) into an array of shape (nf, 2*m),
            where each row consists of all possible ordered pairs of rows from the input array.

            Args:
                x (numpy.ndarray): An input array of shape (n, m), where n is the number of samples
                                and m is the number of features.
                y (numpy.ndarray, optional): An optional input array of shape (n2, m), where n2 is
                                            the number of samples
                                            and m is the number of features. If None, y is set to
                                            x. Defaults to None.
                evaluate_duplicates (str): String indicating which kernel values to evaluate.
                    Options are:
                                        - "off_diagonal": Evaluate only off-diagonal elements.
                                        - "none": Evaluate no duplicates.
                                        - "all": Evaluate all kernel values. Defaults to "all".

            Returns:
                numpy.ndarray: An array of shape (nf, 2*m) where each row consists of all possible
                            ordered pairs of rows
                            from the input array. The value of nf depends on the
                            `evaluate_duplicates` parameter:
                            - If `evaluate_duplicates` is "off_diagonal", nf = n*(n-1).
                            - If `evaluate_duplicates` is "none", nf = n*n-n.
                            - Otherwise, nf = n*n.


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
            if np.array_equal(x, y) and np.allclose(x, y):
                are_equal = True
            else:
                are_equal = False
            if are_equal:
                n = x.shape[0]
                x_rep = np.repeat(x, n, axis=0)  # Repeat each row n times
                x_tile = np.tile(x, (n, 1))  # Tile the entire array n times
            else:
                n = x.shape[0]
                n2 = y.shape[0]
                x_rep = np.repeat(x, n2, axis=0)
                x_tile = np.tile(y, (n, 1))
            result = np.hstack((x_rep, x_tile))

            if are_equal:
                if evaluate_duplicates == "off_diagonal":
                    result = np.delete(
                        result, get_flattened_matrix_indices(n, part="diagonal"), axis=0
                    )
                elif evaluate_duplicates == "none":
                    result = np.delete(
                        result,
                        np.concatenate(
                            [
                                get_flattened_matrix_indices(n, part="higher"),
                                get_flattened_matrix_indices(n, part="diagonal"),
                            ]
                        ),
                        axis=0,
                    )
                elif evaluate_duplicates == "all":
                    pass
                else:
                    raise ValueError(
                        "Invalid evaluate_duplicates option. Use 'off_diagonal', 'none', or 'all'."
                    )
            else:
                self._evaluate_duplicates = "all"
                print("Warning: evaluate_duplicates is set to 'all' as x and y are not equal.")
            return result

        def fill_matrix_indices(K_flat, n, matrix_part):
            """
            Given a flattened kernel matrix of shape (nf,), fills the values not calculated
            according to the specified missing matrix_part.

            Args:
            K_flat (numpy.ndarray):
                Flattened kernel matrix of shape (nf,) where nf is the number of kernel values to
                evaluate.
            n (int): Number of samples in the dataset.
            matrix_part (str): Part of the matrix to fill. Options are "lower" for the
                lower triangle and "diagonal" for the diagonal elements.
                if matrix_part is "lower", K_flat is expected to be of size n*(n-1)/2.
                if matrix_part is "diagonal", K_flat is expected to be of size n*n-n

            Returns:
                numpy.ndarray: Filled kernel matrix in flattened form of shape (n*n,).

            """
            # Fill the upper triangle from the lower triangle
            K_flat_filled = np.zeros(n * n)

            main_diagonal_indices = get_flattened_matrix_indices(n, part="diagonal")
            K_flat_filled[main_diagonal_indices] = 1.0  # Set diagonal elements to 1
            if matrix_part == "lower":
                lower_triangle_indices = get_flattened_matrix_indices(n, part="lower")
                K_flat_filled[lower_triangle_indices] = (
                    K_flat  # Fill the lower triangle with the kernel values
                )
            elif matrix_part == "diagonal":
                non_diagonal_indices = get_flattened_matrix_indices(n, part="off_diagonal")
                print(non_diagonal_indices)
                K_flat_filled[non_diagonal_indices] = K_flat
            else:
                raise ValueError("Invalid matrix_part specified. Use 'lower' or 'diagonal'.")
            return K_flat_filled

        def reshape_to_kernel_matrix(k_flat, n, n2, evaluate_duplicates):
            """
            Reshapes the flattened kernel matrix to a 2D kernel matrix.

            Args:
                k_flat (numpy.ndarray):
                    Flattened kernel matrix of shape (n*n,) or (n*n2, ) where n is the number of
                    samples in the dataset.
                n (int): Number of samples in the dataset.
                n2 (int): Number of samples in the dataset.
                evaluate_duplicates (str): String indicating which kernel values to evaluate.
                    Options are
                    "off_diagonal" to evaluate only off-diagonal elements
                    "none" to evaluate no duplicates,
                    "all" to evaluate all kernel values.
            Returns:
                numpy.ndarray: Kernel matrix of shape (n, n) or (n, n2) depending on the
                input shape.
            """

            if n == n2:
                if evaluate_duplicates == "off_diagonal":
                    k_flat = fill_matrix_indices(k_flat, n, "diagonal")
                    k_flat = k_flat.reshape(n, n)
                elif evaluate_duplicates == "none":
                    k_flat = fill_matrix_indices(k_flat, n, "lower")
                    k_flat = k_flat.reshape(n, n)
                    # make symmetric
                    k_flat = k_flat + k_flat.T - np.diag(k_flat.diagonal())
                elif evaluate_duplicates == "all":
                    k_flat = k_flat.reshape(n, n)
            else:
                k_flat = k_flat.reshape(n, n2)

            return k_flat

        if y is None:
            y = x

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
        # _qnn that implements a circuit U(y)^\dagger U(x) |0>
        # and then expectation value of P0=|0><0|^\otimes n, such that tr(\rho(x), \rho(y))
        # is obtained.
        # we use squlearn's circuit compose and inverse
        self._qnn = LowLevelQNN(
            (self.encoding_circuit).compose(self.encoding_circuit.inverse()),
            P0_operator(self.encoding_circuit.num_qubits),
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

        value_dict["x"] = to_FQK_circuit_format(x, y, "all")
        if self._evaluate_duplicates != "all":
            value_dict["x_without_duplicates"] = to_FQK_circuit_format(
                x, y, self._evaluate_duplicates
            )

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
                if self._evaluate_duplicates != "all" and todo != "K":
                    print(
                        f"Warning: evaluate_duplicates is set to {self._evaluate_duplicates} but, "
                        f"evaluate_duplicates = {self._evaluate_duplicates} is not yet supported"
                        f"for{todo}. evaluate_duplicates='all' will be used for this evaluation."
                    )
                if todo == "K":
                    if self._evaluate_duplicates == "all":
                        kernel_matrix = eval_helper(value_dict["x"], "f")
                    else:
                        kernel_matrix = eval_helper(value_dict["x_without_duplicates"], "f")
                    kernel_matrix = reshape_to_kernel_matrix(
                        kernel_matrix, x.shape[0], y.shape[0], self._evaluate_duplicates
                    )
                elif todo == "dKdx" or todo == "dKdy":
                    dKdx = eval_helper(value_dict["x"], "dfdx").reshape(
                        x.shape[0], y.shape[0], 2 * self.num_features
                    )  # shape (len(x), len(y), 2*num_features)
                    # For consistency with the PQK derivatives:
                    # we transpose the dKdx matrix to be of shape (2*num_features, len(x), len(y))
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
                    # For consistency with the PQK derivatives:
                    # we transpose the dKdp matrix to be of shape (num_parameters, len(x), len(y))
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
                    # For consistency with the PQK derivatives:
                    # we transpose the jacobian matrix to be of shape
                    # (2*num_features, 2*num_features, len(x), len(y))
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
                else:
                    raise NotImplementedError(f"Derivative {todo} not implemented.")

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
