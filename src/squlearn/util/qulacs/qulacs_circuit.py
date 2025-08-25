import numpy as np
from typing import List, Union, Iterable
from sympy import lambdify, sympify

from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.quantum_info import SparsePauliOp
from qiskit.compiler import transpile

from qulacs import ParametricQuantumCircuit, QuantumCircuit
from qulacs import GeneralQuantumOperator, PauliOperator

from .qulacs_gates import qiskit_qulacs_gate_dict, qiskit_qulacs_param_gate_dict
from ..decompose_to_std import decompose_to_std


class QulacsCircuit:
    """
    Class for converting a Qiskit circuit to a Qulacs circuit.

    Args:
        circuit (QiskitQuantumCircuit): Qiskit circuit to convert to Qulacs
        observable (Union[None, SparsePauliOp, List[SparsePauliOp]]): Observable to be measured,
            if None, no observable is defined.

    Attributes:
    -----------

    Attributes:
        num_qubits (int): Number of qubits in the circuit
        circuit_parameter_names (list): List of circuit parameter names
        observable_parameter_names (list): List of observable parameter names
        argument_names (list): List of all circuit and observable parameters names
        hash (str): Hashable object of the circuit and observable for caching

    Methods:
    --------
    """

    def __init__(
        self,
        circuit: QiskitQuantumCircuit,
        observable: Union[
            None,
            SparsePauliOp,
            List[SparsePauliOp],
            str,
        ] = None,
    ) -> None:

        # Transpile circuit to supported basis gates and expand blocks automatically
        self._qiskit_circuit = transpile(
            decompose_to_std(circuit),
            basis_gates=qiskit_qulacs_gate_dict.keys(),
            optimization_level=0,
        )
        self._qiskit_observable = observable
        self._cache_key = str(self._qiskit_circuit) + str(self._qiskit_observable)
        self._num_qubits = self._qiskit_circuit.num_qubits

        self._is_qiskit_observable = False
        if isinstance(observable, SparsePauliOp):
            self._is_qiskit_observable = True
        if isinstance(observable, list):
            if all([isinstance(obs, SparsePauliOp) for obs in observable]):
                self._is_qiskit_observable = True
            else:
                raise ValueError("Only lists of SparsePauliOps are supported")

        # Build qulacs circuit from the qiskit circuit
        self._build_circuit_instructions(self._qiskit_circuit)

        # Build observable instructions
        if self._is_qiskit_observable:
            self.build_observable_instructions(self._qiskit_observable)

        self._circuit_func_cache = {}
        self._observable_func_cache = None
        self._outer_jacobi_circuit_cache = {}
        self._outer_jacobi_observable_cache = {}

    @property
    def num_qubits(self) -> int:
        """Number of qubits of the circuit"""
        return self._num_qubits

    @property
    def circuit_parameter_names(self) -> list:
        """List of circuit parameter names"""
        return self._circuit_param_names

    @property
    def observable_parameter_names(self) -> list:
        """List of observable parameter names"""
        return self._observable_param_names

    @property
    def argument_names(self) -> list:
        """List of all circuit and observable parameters names"""
        return self._circuit_param_names + self._observable_param_names

    @property
    def hash(self) -> str:
        """Hashable object of the circuit and observable for caching"""
        return self._cache_key

    def _add_parameter_expression(
        self, angle: Union[ParameterVectorElement, ParameterExpression, float]
    ) -> tuple:
        """
        Adds a parameter expression (angle) and generate the functions for the gradient calculation

        Args:
            angle (Union[ParameterVectorElement, ParameterExpression, float]): angle of rotation

        Returns:
            tuple: A tuple containing the function to evaluate the angle, the function to
                evaluate the gradient, the used parameters, and a boolean indicating if the
                angle is parameterized at all.
        """

        func_list_element = None
        func_grad_list_element = None
        used_parameters = []

        # Important: change the sign because of the way Qulacs defines the rotation gates
        angle = -angle

        if isinstance(angle, float):
            # Single float value, not parameterized
            func_list_element = angle
            func_grad_list_element = None

        elif isinstance(angle, ParameterVectorElement):
            # Single parameter vector element, no expression
            func_list_element = lambdify(self._circuit_symbols_tuple, sympify(angle._symbol_expr))
            func_grad_list_element = [lambda x: 1.0]
            used_parameters = [angle]

        elif isinstance(angle, ParameterExpression):
            # Parameter is in a expression (equation)
            func_list_element = lambdify(self._circuit_symbols_tuple, sympify(angle._symbol_expr))
            func_grad_list_element = []
            # loop over the parameters in the expression
            for param_element in angle._parameter_symbols.keys():
                used_parameters.append(param_element)
                # get the gradient of the parameter expression wrt the parameter
                param_grad = angle.gradient(param_element)
                if isinstance(param_grad, float):
                    # create a call by value labmda function
                    func_grad_list_element.append(lambda *arg, param_grad=param_grad: param_grad)
                else:
                    func_grad_list_element.append(
                        lambdify(self._circuit_symbols_tuple, sympify(param_grad._symbol_expr))
                    )
        else:
            raise ValueError("Unsupported angle type")

        return func_list_element, func_grad_list_element, used_parameters

    def _add_single_qubit_gate(self, gate_name: str, qubits: Union[int, Iterable[int]]):
        """
        Adds a single qubit, non-parameterized gate to the circuit.

        Adds gates to multiple qubits if qubits is an iterable.

        Args:
            gate_name (str): Name of the gate
            qubits (int or Iterable[int]): qubit indices
        """
        qubits = [qubits] if isinstance(qubits, int) else qubits
        for q in qubits:
            if q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} is out of range")
            self._circuit_gate_list.append(gate_name)
            self._circuit_qubit_list.append([q])
            self._circuit_param_func_list.append(None)
            self._circuit_param_func_grad_list.append(None)
            self._circuit_param_list.append([])

    def _add_two_qubit_gate(
        self, gate_name: str, qubit1: Union[int, Iterable[int]], qubit2: Union[int, Iterable[int]]
    ) -> None:
        """
        Adds a two qubit, non-parameterized gate to the circuit.

        Adds gates to multiple qubits if qubit1 and qubit2 are both iterables.

        Args:
            gate_name (str): Name of the gate
            qubit1 (int or Iterable[int]): qubit indices of the first qubit (e.g. control)
            qubit2 (int or Iterable[int]): qubit indices of the second qubit (e.g. target)
        """
        qubit1 = [qubit1] if isinstance(qubit1, int) else qubit1
        qubit2 = [qubit2] if isinstance(qubit2, int) else qubit2

        for control, target in zip(qubit1, qubit2):
            if control >= self.num_qubits or target >= self.num_qubits:
                raise ValueError(f"Qubit index is out of range")
            self._circuit_gate_list.append(gate_name)
            self._circuit_qubit_list.append([control, target])
            self._circuit_param_func_list.append(None)
            self._circuit_param_func_grad_list.append(None)
            self._circuit_param_list.append([])

    def _add_parameterized_single_qubit_gate(
        self,
        gate_name: str,
        qubits: Union[int, Iterable[int]],
        angle: Union[ParameterVectorElement, ParameterExpression, float],
    ):
        """
        Adds a single qubit, parameterized gate to the circuit.

        Adds gates to multiple qubits if qubits is an iterable.

        Args:
            gate_name (str): Name of the gate
            qubits (Union[int, Iterable[int]]): qubit indices
            angle (Union[ParameterVectorElement, ParameterExpression, float]): angle of rotation
        """

        # pre-process the angle in case it is a parameter expression
        func_list_element, func_grad_list_element, used_parameters = (
            self._add_parameter_expression(angle)
        )

        qubits = [qubits] if isinstance(qubits, int) else qubits
        for q in qubits:
            if q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} is out of range")
            self._circuit_gate_list.append(gate_name)
            self._circuit_qubit_list.append([q])
            self._circuit_param_func_list.append(func_list_element)
            self._circuit_param_func_grad_list.append(func_grad_list_element)
            self._circuit_param_list.append(used_parameters)

    def _add_parameterized_two_qubit_gate(
        self,
        gate_name: str,
        qubit1: Union[int, Iterable[int]],
        qubit2: Union[int, Iterable[int]],
        angle: Union[ParameterVectorElement, ParameterExpression, float],
    ):
        """
        Adds a single qubit parameterized gate to the circuit.

        Adds gates to multiple qubits if qubit1 and qubit2 are both iterables.

        Args:
            gate_name (str): Name of the gate
            qubit1 (Union[int, Iterable[int]]): qubit indices of the first qubit (e.g. control
            qubit2 (Union[int, Iterable[int]]): qubit indices of the second qubit (e.g. target)
            angle (Union[ParameterVectorElement, ParameterExpression, float]): angle of rotation
        """
        func_list_element, func_grad_list_element, used_parameters = (
            self._add_parameter_expression(angle)
        )

        qubit1 = [qubit1] if isinstance(qubit1, int) else qubit1
        qubit2 = [qubit2] if isinstance(qubit2, int) else qubit2

        for control, target in zip(qubit1, qubit2):
            if control >= self.num_qubits or target >= self.num_qubits:
                raise ValueError(f"Qubit index is out of range")
            self._circuit_gate_list.append(gate_name)
            self._circuit_qubit_list.append([control, target])
            self._circuit_param_func_list.append(func_list_element)
            self._circuit_param_func_grad_list.append(func_grad_list_element)
            self._circuit_param_list.append(used_parameters)

    def _build_circuit_instructions(self, circuit: QiskitQuantumCircuit) -> tuple:
        """
        Function to build the instructions for the Qulacs circuit from the Qiskit circuit.

        This functions converts the Qiskit gates and parameter expressions to Qulacs compatible
        gates and functions.

        Args:
            circuit (QiskitQuantumCircuit): Qiskit circuit to convert to Qulacs

        """
        # Data structures to store the circuit information
        # ------------------------------------------------
        # Gate names that correspond to qulacs_gates.py
        self._circuit_gate_list = []
        # Qubit indices for each gate
        self._circuit_qubit_list = []
        # List of used parameters for each gate
        self._circuit_param_list = []
        # Functions to evaluate the angles for each gate
        self._circuit_param_func_list = []
        # Functions to evaluate the gradients of the angle for each gate
        self._circuit_param_func_grad_list = []
        # List of parameter names used in the circuit
        self._circuit_param_names = []
        # Tuple of parameter vector elements used in the circuit for symbolice valuation
        self._circuit_symbols_tuple = []

        # Create the circuit parameter names and symbols tuple from all parameters of the circuit
        for param in circuit.parameters:
            if param.vector.name not in self._circuit_param_names:
                self._circuit_param_names.append(param.vector.name)
                self._circuit_symbols_tuple += [
                    sympify(p._symbol_expr) for p in param.vector.params
                ]

        # Sort the symbols tuple by variable name and index
        def sort_key(item):
            item_string = repr(item)
            variable_name = item_string.split("[")[0]
            index = int(item_string[item_string.index("[") + 1 : item_string.index("]")])
            return (variable_name, index)

        self._circuit_symbols_tuple = tuple(sorted(self._circuit_symbols_tuple, key=sort_key))

        # Loop over the circuit data and convert the gates to the qulacs circuit format
        for op in circuit.data:

            # mid-circuit measurements and conditions are not supported!
            if op.operation.condition is not None or op.operation.name == "measure":
                raise NotImplementedError(
                    "Conditions are not supported in sQUlearn's Qulacs backend."
                )

            # Catch unsupported gates (should not happen due to transpilation)
            if (
                op.operation.name not in qiskit_qulacs_gate_dict
                and op.operation.name not in qiskit_qulacs_param_gate_dict
            ):
                raise NotImplementedError(
                    f"Gate {op.operation.name} is unfortunatly not supported in "
                    "sQUlearn's Qulacs backend."
                )

            # Get the qubit indices for the operation
            qubits_indices = [
                circuit.find_bit(op.qubits[i]).index for i in range(op.operation.num_qubits)
            ]

            # Check if the operation is parameterized and single or two qubit
            is_parameterized = len(op.operation.params) >= 1
            is_single_qubit = len(op.qubits) == 1

            if is_single_qubit:
                if not is_parameterized:
                    self._add_single_qubit_gate(op.operation.name, qubits_indices)
                else:
                    self._add_parameterized_single_qubit_gate(
                        op.operation.name, qubits_indices, op.operation.params[0]
                    )
            else:
                if len(op.qubits) > 2:
                    raise NotImplementedError(
                        "Only two qubit gates are supported in sQUlearn's Qulacs backend."
                    )
                if not is_parameterized:
                    self._add_two_qubit_gate(
                        op.operation.name, qubits_indices[0], qubits_indices[1]
                    )
                else:
                    self._add_parameterized_two_qubit_gate(
                        op.operation.name,
                        qubits_indices[0],
                        qubits_indices[1],
                        op.operation.params[0],
                    )

    def build_observable_instructions(
        self, observables: Union[List[SparsePauliOp], SparsePauliOp]
    ):
        """
        Function to build the instructions for the Qulacs observable from the Qiskit observable.

        This functions converts the Qiskit SparsePauli and parameter expressions to Qulacs
        compatible Pauli words and functions.

        Args:
            observable (Union[List[SparsePauliOp], SparsePauliOp]): Qiskit observable to convert
                                                                    to Qulacs

        Returns:
            Tuple with lists of Qulacs observable parameter functions, Qulacs Pauli words,
            Qulacs observable parameters and Qulacs observable parameter dimensions
        """

        self.multiple_observables = False
        if isinstance(observables, SparsePauliOp):
            observables = [observables]
        elif isinstance(observables, list):
            self.multiple_observables = True
        else:
            raise ValueError("Unsupported observable type")

        self._observable_pauli_list = []
        self._observable_param_func_list = []
        self._observable_param_func_grad_list = []
        self._observable_used_parameters = []
        self._observable_symbols_tuple = tuple()
        self._observable_param_names = []

        # Create the observable parameter names and symbols tuple from all parameters of the
        # observables
        for observable in observables:
            for param in observable.parameters:
                if param.vector.name not in self._observable_param_names:
                    self._observable_param_names.append(param.vector.name)

        # Sort the symbols tuple by variable name and index
        def sort_parameters_after_index(parameter_vector):
            index_list = [p.index for p in parameter_vector]
            argsort_list = np.argsort(index_list)
            return [parameter_vector[i] for i in argsort_list]

        self._observable_symbols_tuple = tuple(
            sum(
                [
                    [sympify(p._symbol_expr) for p in sort_parameters_after_index(obs.parameters)]
                    for obs in observables
                ],
                [],
            )
        )

        # Loop over the observables and convert the Pauli words to the qulacs observable format
        for observable in observables:

            # Extract the Pauli strings and coefficients from the SparsePauliOp
            # reverse the Pauli word to match Qulacs convention
            paulis = [str(p[::-1]) for p in observable._pauli_list]
            coefficients = list(np.real_if_close([c for c in observable.coeffs]))

            # Loop over the Pauli strings and coefficients
            observable_paulis = []
            observable_coeff = []
            observable_coeff_grad = []
            observable_used_parameters = []
            for c, p in zip(coefficients, paulis):
                # Create pauli word string for Qulacs (e.g. "X 0 Y 1 Z 2")
                observable_paulis.append(" ".join(f"{p_} {i}" for i, p_ in enumerate(p)))

                if isinstance(c, ParameterVectorElement):
                    # Single parameter vector element
                    observable_coeff.append(
                        lambdify(self._observable_symbols_tuple, sympify(c._symbol_expr))
                    )
                    observable_coeff_grad.append([lambda *arg: 1.0])
                    observable_used_parameters.append([c])

                elif isinstance(c, ParameterExpression):
                    # Parameter is in a expression (equation)
                    observable_coeff.append(
                        lambdify(self._observable_symbols_tuple, sympify(c._symbol_expr))
                    )
                    func_grad_list_element = []
                    used_parameters_obs_element = []
                    for param_element in c._parameter_symbols.keys():
                        used_parameters_obs_element.append(param_element)
                        # information about the gradient of the parameter expression
                        # the 1j fixes a bug in qiskit
                        param_grad = -1j * ((1j * c).gradient(param_element))
                        if isinstance(param_grad, complex):
                            if param_grad.imag == 0:
                                param_grad = param_grad.real
                        if isinstance(param_grad, float) or isinstance(param_grad, complex):
                            # create a call by value labmda function
                            func_grad_list_element.append(
                                lambda *arg, param_grad=param_grad: param_grad
                            )
                        else:
                            func_grad_list_element.append(
                                lambdify(
                                    self._observable_symbols_tuple,
                                    sympify(param_grad._symbol_expr),
                                )
                            )
                    observable_coeff_grad.append(func_grad_list_element)
                    observable_used_parameters.append(used_parameters_obs_element)

                else:
                    # Fixed value, not parameterized
                    observable_coeff.append(lambda *arg, c=c: c)
                    observable_coeff_grad.append([lambda *arg: 0.0])
                    observable_used_parameters.append([])

            self._observable_pauli_list.append(observable_paulis)
            self._observable_param_func_list.append(observable_coeff)
            self._observable_param_func_grad_list.append(observable_coeff_grad)
            self._observable_used_parameters.append(observable_used_parameters)

    def get_observable_func(self):
        """Returns a function for creating the qulacs observable depending on the parameters."""

        if self._observable_func_cache is not None:
            return self._observable_func_cache

        def observable_func(*args):
            """
            Function for building the Qulacs observable from the parameters.

            Args:
                *args: Parameters to evaluate the observable
            """
            list_operators = []
            for i, observable in enumerate(self._observable_pauli_list):
                operator = GeneralQuantumOperator(self.num_qubits)
                for j, op in enumerate(observable):
                    operator.add_operator(self._observable_param_func_list[i][j](*args), op)
                list_operators.append(operator)

            return list_operators

        self._observable_func_cache = observable_func

        return observable_func

    def get_circuit_func(
        self,
        gradient_param: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None,
    ) -> callable:
        """Returns the function for the creating the qulacs circuit and its gradients.

        Args:
            gradient_param (Union[None, ParameterVectorElement, List[ParameterVectorElement]]):
                Parameters to calculate the gradient for

        """

        # Pre-process the gradient parameters
        if isinstance(gradient_param, ParameterVectorElement):
            gradient_param = [gradient_param]
        gradient_param = list(gradient_param) if gradient_param is not None else []
        is_parameterized = len(gradient_param)
        # Identify which operations are have to be parameterized in the qulacs circuit
        parameterized_operations = [
            any(param in gradient_param for param in self._circuit_param_list[i])
            for i, _ in enumerate(self._circuit_gate_list)
        ]

        # set-up the cache for the circuit function especially for the gradient
        cache_value = "no_gradient"
        if is_parameterized:
            cache_value = tuple(gradient_param)
        if cache_value in self._circuit_func_cache:
            return self._circuit_func_cache[cache_value]

        def qulacs_circuit(*args) -> Union[ParametricQuantumCircuit, QuantumCircuit]:
            """Function for building the Qulacs circuit from the numerical parameters values.

            Args:
                *args: Parameters to evaluate the circuit in the order of the circuit parameter
                    names

            Returns:
                Union[ParametricQuantumCircuit, QuantumCircuit]: Qulacs circuit with the gates
                    applied and the parameters evaluated.
            """

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum(
                [list(args[i]) for i in range(len(self._circuit_param_names))], []
            )

            # Create the Qulacs circuit, the ParametricQuantumCircuit is used if the circuit
            # is parameterized to enable fast gradient evaluation in Qulacs
            if is_parameterized:
                circuit = ParametricQuantumCircuit(self.num_qubits)
            else:
                circuit = QuantumCircuit(self.num_qubits)

            # Build the Qulacs circuit and evaluate the parametric terms
            for i, op in enumerate(self._circuit_gate_list):
                if self._circuit_param_func_list[i] is None:
                    qiskit_qulacs_gate_dict[op](circuit, *self._circuit_qubit_list[i])
                elif isinstance(self._circuit_param_func_list[i], float):
                    qiskit_qulacs_gate_dict[op](
                        circuit, self._circuit_param_func_list[i], *self._circuit_qubit_list[i]
                    )
                else:
                    value = self._circuit_param_func_list[i](*circ_param_list)
                    if parameterized_operations[i]:
                        qiskit_qulacs_param_gate_dict[op](
                            circuit, value, *self._circuit_qubit_list[i]
                        )
                    else:
                        qiskit_qulacs_gate_dict[op](circuit, value, *self._circuit_qubit_list[i])

            return circuit

        # Cache the circuit function
        self._circuit_func_cache[cache_value] = qulacs_circuit

        return qulacs_circuit

    def get_outer_jacobian_circuit(
        self,
        gradient_parameters: Union[
            None, ParameterVectorElement, List[ParameterVectorElement]
        ] = None,
    ) -> callable:
        """Returns the outer jacobian needed for the chain rule in circuit derivatives.

        Qulacs does not support multiple parameters and parameter expressions,
        so we need to calculate a transformation which also includes the gradient of the
        parameter expression.

        Args:
            gradient_parameters (Union[None, ParameterVectorElement, List[ParameterVectorElement]]):
                Parameters to calculate the gradient for

        Returns:
            callable: Function that calculates the outer jacobian for given parameters values
        """

        if isinstance(gradient_parameters, ParameterVectorElement):
            gradient_parameters = [gradient_parameters]
        gradient_parameters = list(gradient_parameters) if gradient_parameters is not None else []
        gradient_param_dict = {p: i for i, p in enumerate(gradient_parameters)}

        cache_value = "no_gradient"
        if len(gradient_parameters) > 0:
            cache_value = tuple(gradient_parameters)

        if cache_value in self._outer_jacobi_circuit_cache:
            return self._outer_jacobi_circuit_cache[cache_value]

        def outer_jacobian(*args) -> np.ndarray:
            """Function for calculating the outer jacobian for the circuit parameters.

            Args:
                *args: Parameters to evaluate the circuit in the order of the circuit parameter
                    names

            Returns:
                np.ndarray: Outer jacobian matrix as numpy array
            """

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum(
                [list(args[i]) for i in range(len(self._circuit_param_names))], []
            )

            # Identify which operations are relevant for the gradient
            relevant_operations = [
                i
                for i in range(len(self._circuit_gate_list))
                if any(param in gradient_parameters for param in self._circuit_param_list[i])
            ]

            # Build the outer jacobian matrix
            outer_jacobian = np.zeros((len(relevant_operations), len(gradient_parameters)))
            for i, operation in enumerate(relevant_operations):
                for j, param in enumerate(self._circuit_param_list[operation]):
                    if param in gradient_parameters:
                        outer_jacobian[i, gradient_param_dict[param]] = (
                            self._circuit_param_func_grad_list[operation][j](*circ_param_list)
                        )

            return outer_jacobian

        self._outer_jacobi_circuit_cache[cache_value] = outer_jacobian

        return outer_jacobian

    def get_outer_jacobian_observables(
        self,
        gradient_parameters: Union[
            None, ParameterVectorElement, List[ParameterVectorElement]
        ] = None,
    ) -> callable:
        """Returns the outer jacobian needed for the chain rule in circuit derivatives.

        Qulacs does not support multiple parameters and parameter expressions,
        so we need to calculate a transformation which also includes the gradient of the
        parameter expression.

        Args:
            gradient_parameters (Union[None, ParameterVectorElement, List[ParameterVectorElement]]):
                Parameters to calculate the gradient for

        Returns:
            callable: Function that calculates the outer jacobian for the observable parameters
        """

        # Pre-process the gradient parameters
        if isinstance(gradient_parameters, ParameterVectorElement):
            gradient_parameters = [gradient_parameters]
        gradient_parameters = list(gradient_parameters) if gradient_parameters is not None else []
        gradient_param_dict = {p: i for i, p in enumerate(gradient_parameters)}

        #  Set-up the cache for the observable jacobian function especially for the gradient
        cache_value = "no_gradient"
        if len(gradient_parameters) > 0:
            cache_value = tuple(gradient_parameters)
        if cache_value in self._outer_jacobi_observable_cache:
            return self._outer_jacobi_observable_cache[cache_value]

        def outer_jacobian(*args) -> List[np.ndarray]:
            """Function for calculating the outer jacobian for the observable parameters.

            Args:
                *args: Parameters to evaluate the observable in the order of the observable
                    parameter names

            Returns:
                List[np.ndarray]: List of outer jacobian matrices for each observable
                    as numpy arrays
            """

            # Collects the args values connected to the observable parameters
            obs_param_list = sum(
                [list(args[i]) for i in range(len(self._observable_param_names))], []
            )

            outer_jacobians = []
            for iop, operator in enumerate(self._observable_param_func_grad_list):

                relevant_operations = [
                    i
                    for i in range(len(operator))
                    if any(
                        param in gradient_parameters
                        for param in self._observable_used_parameters[iop][i]
                    )
                ]

                outer_jacobian = np.zeros((len(relevant_operations), len(gradient_parameters)))
                for i, operation in enumerate(relevant_operations):
                    for j, param in enumerate(self._observable_used_parameters[iop][operation]):
                        if param in gradient_parameters:
                            outer_jacobian[i, gradient_param_dict[param]] = (
                                self._observable_param_func_grad_list[iop][operation][j](
                                    *obs_param_list
                                )
                            )
                outer_jacobians.append(outer_jacobian)
            return outer_jacobians

        # Cache the outer jacobian function
        self._outer_jacobi_observable_cache[cache_value] = outer_jacobian

        return outer_jacobian

    def get_observables_for_gradient(
        self,
        gradient_parameters: Union[
            None, ParameterVectorElement, List[ParameterVectorElement]
        ] = None,
    ) -> callable:
        """Returns the Qulacs observable function for the observable depending on parameters.

        Args:
            gradient_parameters (Union[None, ParameterVectorElement, List[ParameterVectorElement]]):
                Parameters to calculate the gradient for

        Returns:
            callable: Function that returns the Qulacs observable for the given parameters
        """

        # Pre-process the gradient parameters
        if isinstance(gradient_parameters, ParameterVectorElement):
            gradient_parameters = [gradient_parameters]
        gradient_parameters = list(gradient_parameters) if gradient_parameters is not None else []

        def observable_func(*args) -> List[List[PauliOperator]]:
            """
            Function for building the Qulacs observable used in the gradient evaluation.

            Args:
                args: Parameters to evaluate the observable in the order of the observable
                    parameter names

            Returns:
                List[List[PauliOperator]]: Nested list of Qulacs Pauli operators for the
                    observables

            """

            list_operators = []
            # Build lists of the single Pauli operators for each observable
            for iop, observable in enumerate(self._observable_pauli_list):
                relevant_operations = [
                    i
                    for i in range(len(observable))
                    if any(
                        param in gradient_parameters
                        for param in self._observable_used_parameters[iop][i]
                    )
                ]
                list_paulis = []
                for op in relevant_operations:
                    list_paulis.append(PauliOperator(observable[op], 1.0))
                list_operators.append(list_paulis)

            return list_operators

        return observable_func

    def __getstate__(self):
        """Remove qulacs caches and objects to avoid pickling issues"""
        state = self.__dict__.copy()
        # Remove caches to avoid pickling issues
        state["_circuit_func_cache"] = {}
        state["_observable_func_cache"] = None
        state["_outer_jacobi_circuit_cache"] = {}
        state["_outer_jacobi_observable_cache"] = {}

        # Remove qulacs instructions to avoid pickling issues
        state["_circuit_gate_list"] = []
        state["_circuit_qubit_list"] = []
        state["_circuit_param_list"] = []
        state["_circuit_param_func_list"] = []
        state["_circuit_param_func_grad_list"] = []
        state["_circuit_param_names"] = []
        state["_circuit_symbols_tuple"] = ()
        state["_observable_pauli_list"] = []
        state["_observable_param_func_list"] = []
        state["_observable_param_func_grad_list"] = []
        state["_observable_used_parameters"] = []
        state["_observable_symbols_tuple"] = ()
        state["_observable_param_names"] = []

        return state

    def __setstate__(self, state):
        """Rebuild the circuit and observable instructions from the state"""
        self.__dict__.update(state)
        # Rebuild circuit instructions
        self._build_circuit_instructions(self._qiskit_circuit)
        # Rebuild observable instructions if needed
        if self._is_qiskit_observable:
            self.build_observable_instructions(self._qiskit_observable)
