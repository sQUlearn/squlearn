import numpy as np
from typing import Union, List
from sympy import lambdify, sympify

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.classicalregister import Clbit

from qulacs import QuantumCircuit, QuantumState, QuantumCircuitSimulator, CausalConeSimulator

from typing import List, Union, Iterable

from qiskit.compiler import transpile
from qiskit_aer import Aer

from qulacs import ParametricQuantumCircuit
from qulacs import Observable
#import qulacs as qml
#import qulacs.numpy as pnp
#import qulacs.pauli as pauli
#from qulacs.operation import Observable as QulacsObservable

from .qulacs_gates import qiskit_qulacs_gate_dict, qiskit_qulacs_param_gate_dict
from ..executor import Executor
from ..decompose_to_std import decompose_to_std


class QulacsCircuit:
    """
    Class for converting a Qiskit circuit to a Qulacs circuit.

    Args:
        circuit (QuantumCircuit): Qiskit circuit to convert to Qulacs
        observable (Union[None, SparsePauliOp, List[SparsePauliOp], str]): Observable to be measured
                                                                           Can be also a string like ``"probs"`` or ``"state"``
        executor (Executor): Executor object to handle the Qulacs circuit. Has to be initialized with a Qulacs device.

    Attributes:
    -----------

    Attributes:
        qulacs_circuit (qml.qnode): Qulacs circuit that can be called with parameters
        circuit_parameter_names (list): List of circuit parameter names
        observable_parameter_names (list): List of observable parameter names
        circuit_parameter_dimensions (dict): Dictionary with the dimension of each circuit parameter
        observable_parameter_dimension (dict): Dictionary with the dimension of each observable parameter
        circuit_arguments (list): List of all circuit and observable parameters names
        hash (str): Hashable object of the circuit and observable for caching

    Methods:
    --------
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        observable: Union[
            None,
            SparsePauliOp,
            List[SparsePauliOp],
            str,
            #QulacsObservable, TODO?
            #List[QulacsObservable], TODO?
        ] = None,
        executor: Executor = None,
    ) -> None:

        self._executor = executor
        if self._executor is None:
            pass
            #self._executor = Executor("qulacs") #  TODO implement

        # Transpile circuit to supported basis gates and expand blocks automatically
        self._qiskit_circuit = transpile(
            decompose_to_std(circuit),
            basis_gates=qiskit_qulacs_gate_dict.keys(),
            optimization_level=0,
        )
        
        #print("self._qiskit_circuit",self._qiskit_circuit)

        #self._qiskit_circuit = decompose_to_std(circuit)
        #self._qiskit_circuit = circuit

        self._qiskit_observable = observable
        self._num_qubits = self._qiskit_circuit.num_qubits


        self._is_qiskit_observable = False
        if isinstance(observable, SparsePauliOp):
            self._is_qiskit_observable = True
        if isinstance(observable, list):
            if all([isinstance(obs, SparsePauliOp) for obs in observable]):
                self._is_qiskit_observable = True

        # Build circuit instructions for the qulacs observable from the qiskit circuit

        self._operation_list = []
        self._qubit_list = []
        self._func_list = []
        self._func_grad_list = []
        self._free_parameters = set()
        
        self._qualcs_gates_parameters = []
        self._symbol_tuple = tuple()
        

        self._rebuild_circuit_func = True
        self._circuit_func = None

        self._num_clbits = self._qiskit_circuit.num_clbits

        self._build_circuit_instructions(self._qiskit_circuit)

        self._operators_imag = []
        self._operators_real = []
        self.build_observable_instructions(self._qiskit_observable)

        #self._qulacs_circuit = self.build_qulacs_circuit()

    @property
    def num_qubits(self) -> int:
        """Number of qubits of the circuit"""
        return self._num_qubits

    @property
    def qulacs_circuit(self) -> callable:
        """Qulacs circuit that can be called with parameters"""
        return self._qulacs_circuit

    @property
    def circuit_parameter_names(self) -> list:
        """List of circuit parameter names"""
        return self._qulacs_gates_parameters

    @property
    def observable_parameter_names(self) -> list:
        """List of observable parameter names"""
        return self._qulacs_obs_parameters

    # @property
    # def circuit_parameter_dimensions(self) -> dict:
    #     """Dictionary with the dimension of each circuit parameter"""
    #     return self._qulacs_gates_parameters_dimensions

    # @property
    # def observable_parameter_dimensions(self) -> dict:
    #     """Dictionary with the dimension of each observable parameter"""
    #     return self._qulacs_obs_parameters_dimensions

    @property
    def circuit_arguments(self) -> list:
        """List of all circuit and observable parameters names"""
        return self._qualcs_gates_parameters + self._qualcs_obs_parameters

    @property
    def hash(self) -> str:
        """Hashable object of the circuit and observable for caching"""
        return str(self._qiskit_circuit) + str(self._qiskit_observable)

    def draw(self, engine: str = "qulacs", **kwargs):
        """Draw the circuit with the specified engine

        Args:
            engine (str): Engine to draw the circuit. Can be either ``"qulacs"`` or ``"qiskit"``
            **kwargs: Additional arguments for the drawing engine (only for qiskit)

        Returns:
            matplotlib Figure object of the circuit visualization
        """

        # Use Qiskit for drawing the circuit

        raise NotImplementedError("Circuit engine not implemented")

    def get_qulacs_circuit(self) -> callable:
        """Builds and returns the Qulacs circuit as callable function"""
        self._qulacs_circuit = self.build_qulacs_circuit()
        return self._qulacs_circuit

    def __call__(self, *args, **kwargs):
        return self._qulacs_circuit(*args, **kwargs)


    def _add_parameter_expression(
        self, angle: Union[ParameterVectorElement, ParameterExpression, float]
    ):
        """
        Adds a parameter expression to the circuit and do the pre-processing.

        Args:
            angle (ParameterVectorElement or ParameterExpression or float): angle of rotation

        Returns:
            int: index of the parameter in the parameter vector
            callable: function to calculate the parameter expression
            callable: function to calculate the gradient of the parameter expression

        """

        # In case angle is a float or something similar, we do not need to add a parameter
        func_list_element = None
        func_grad_list_element = None
        parameterized = False

        # Change sign because of the way Qulacs defines the rotation gates
        angle = -angle

        if isinstance(angle, float):
            # Single float value
            func_list_element = angle
            func_grad_list_element = None
        elif isinstance(angle, ParameterVectorElement):
            # Single parameter vector element
            parameterized = True
            func_list_element = lambdify(self._symbol_tuple, sympify(angle._symbol_expr))
            func_grad_list_element = [lambda x: 1.0]
            self._free_parameters.add(angle)

        elif isinstance(angle, ParameterExpression):
            # Parameter is in a expression (equation)
            parameterized = True
            func_list_element = lambdify(self._symbol_tuple, sympify(angle._symbol_expr))
            func_grad_list_element = []
            for param_element in angle._parameter_symbols.keys():
                self._free_parameters.add(param_element)
                # information about the gradient of the parameter expression
                param_grad = angle.gradient(param_element)
                if isinstance(param_grad, float):
                    # create a call by value labmda function
                    func_grad_list_element.append(lambda *arg, param_grad=param_grad: param_grad)
                else:
                    func_grad_list_element.append(lambdify(self._symbol_tuple, sympify(param_grad._symbol_expr)))

        return func_list_element, func_grad_list_element, parameterized

    def _add_single_qubit_gate(self, gate_name: str, qubits: Union[int, Iterable[int]]):
        """
        Adds a single qubit gate to the circuit.

        Args:
            gate_name (str): Name of the gate
            qubits (int or Iterable[int]): qubit indices
        """
        qubits = [qubits] if isinstance(qubits, int) else qubits
        for q in qubits:
            self._operation_list.append(qiskit_qulacs_gate_dict[gate_name])
            if q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} is out of range")
            self._qubit_list.append([q])
            self._func_list.append(None)
            self._func_grad_list.append(None)

        self._rebuild_circuit_func = True

    def _add_two_qubit_gate(
        self, gate_name: str, qubit1: Union[int, Iterable[int]], qubit2: Union[int, Iterable[int]]
    ) -> None:
        """
        Adds a two qubit gate to the circuit.

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
            self._operation_list.append(qiskit_qulacs_gate_dict[gate_name])
            self._qubit_list.append([control, target])
            self._func_list.append(None)
            self._func_grad_list.append(None)

        self._rebuild_circuit_func = True

    def _add_parameterized_single_qubit_gate(
        self,
        gate_name: str,
        qubits: Union[int, Iterable[int]],
        angle: Union[ParameterVectorElement, ParameterExpression, float],
    ):
        """
        Adds a single qubit parameterized gate to the circuit.

        Args:
            gate_name (str): Name of the gate
            qubits (int or Iterable[int]): qubit indices
            angle (ParameterVectorElement or float): angle of rotation, ca be a parameter
        """
        func_list_element, func_grad_list_element, parameterized = (
            self._add_parameter_expression(angle)
        )

        qubits = [qubits] if isinstance(qubits, int) else qubits
        for q in qubits:
            if q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} is out of range")
            if parameterized:
                self._operation_list.append(qiskit_qulacs_param_gate_dict[gate_name])
            else:
                self._operation_list.append(qiskit_qulacs_gate_dict[gate_name])
            self._qubit_list.append([q])
            self._func_list.append(func_list_element)
            self._func_grad_list.append(func_grad_list_element)

        self._rebuild_circuit_func = True

    def _add_parameterized_two_qubit_gate(
        self,
        gate_name: str,
        qubit1: Union[int, Iterable[int]], qubit2: Union[int, Iterable[int]],
        angle: Union[ParameterVectorElement, ParameterExpression, float],
    ):
        """
        Adds a single qubit parameterized gate to the circuit.

        Args:
            gate_name (str): Name of the gate
            qubits (int or Iterable[int]): qubit indices
            angle (ParameterVectorElement or float): angle of rotation, ca be a parameter
        """
        func_list_element, func_grad_list_element, parameterized = (
            self._add_parameter_expression(angle)
        )

        qubit1 = [qubit1] if isinstance(qubit1, int) else qubit1
        qubit2 = [qubit2] if isinstance(qubit2, int) else qubit2

        for control, target in zip(qubit1, qubit2):
            if control >= self.num_qubits or target >= self.num_qubits:
                raise ValueError(f"Qubit index is out of range")
            if parameterized:
                self._operation_list.append(qiskit_qulacs_param_gate_dict[gate_name])
            else:
                self._operation_list.append(qiskit_qulacs_gate_dict[gate_name])
            self._qubit_list.append([control, target])
            self._func_list.append(func_list_element)
            self._func_grad_list.append(func_grad_list_element)

        self._rebuild_circuit_func = True


    def _build_circuit_instructions(self, circuit: QuantumCircuit) -> tuple:
        """
        Function to build the instructions for the Qulacs circuit from the Qiskit circuit.

        This functions converts the Qiskit gates and parameter expressions to Qulacs compatible
        gates and functions.

        Args:
            circuit (QuantumCircuit): Qiskit circuit to convert to Qulacs

        Returns:
            Tuple with lists of Qulacs gates, Qulacs gate parameter functions,
            Qulacs gate wires, Qulacs gate parameters and Qulacs gate parameter dimensions
        """

        self._operation_list = []
        self._param_list = []
        self._qubit_list = []
        self._func_list = []
        self._func_grad_list = []
        self._free_parameters = set()
        self._qualcs_gates_parameters = []
        self._symbol_tuple = tuple()

        for param in circuit.parameters:
            if param.vector.name not in self._qualcs_gates_parameters:
                self._qualcs_gates_parameters.append(param.vector.name)

        self._symbol_tuple = tuple([sympify(p._symbol_expr) for p in circuit.parameters])

        for op in circuit.data:

            # catch conditions of the gate
            # only c_if is supported, the other cases have been caught before
            if op.operation.condition is not None or op.operation.name == "measure":
                raise NotImplementedError("Conditions are not supported in sQUlearn's Qulacs backend.")

            if op.operation.name not in qiskit_qulacs_gate_dict and op.operation.name not in qiskit_qulacs_param_gate_dict:
                raise NotImplementedError(
                    f"Gate {op.operation.name} is unfortunatly not supported in sQUlearn's Qulacs backend."
                )

            paramterized_gate = len(op.operation.params) >= 1
            single_qubit_date = len(op.qubits) == 1

            wires = [
                circuit.find_bit(op.qubits[i]).index for i in range(op.operation.num_qubits)
            ]

            if single_qubit_date:
                if not paramterized_gate:
                    self._add_single_qubit_gate(op.operation.name, wires)
                else:
                    self._add_parameterized_single_qubit_gate(op.operation.name, wires, op.operation.params[0])
            else:
                if len(op.qubits)>2:
                    raise NotImplementedError("Only two qubit gates are supported in sQUlearn's Qulacs backend.")
                if not paramterized_gate:
                    self._add_two_qubit_gate(op.operation.name, wires[0], wires[1])
                else:
                    self._add_parameterized_two_qubit_gate(op.operation.name, wires[0], wires[1], op.operation.params[0])




    def build_observable_instructions(self, observables: Union[List[SparsePauliOp], SparsePauliOp]):
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
#        if observables == None:
#            return None, None, None

        self.multiple_observables = False
        if isinstance(observables, SparsePauliOp):
            observables = [observables]
        elif isinstance(observables, list):
            self.multiple_observables = True
        else:
            raise ValueError("Unsupported observable type")

        self._symbol_tuple = tuple()

        self._qualcs_obs_parameters = []

        for observable in observables:
            for param in observable.parameters:
                if param.vector.name not in self._qualcs_obs_parameters:
                    self._qualcs_obs_parameters.append(param.vector.name)

        def sort_parameters_after_index(parameter_vector):
            index_list = [p.index for p in parameter_vector]
            argsort_list = np.argsort(index_list)
            return [parameter_vector[i] for i in argsort_list]


        self._symbol_tuple = tuple(
            sum(
                [
                    [sympify(p._symbol_expr) for p in sort_parameters_after_index(obs.parameters)]
                    for obs in observables
                ],
                [],
            )
        )

        # Convert observables
        self._operators_real = []
        self._operators_imag = []
        self._operators_param = []
        self._operators_param_func = []
        self._operators_param_func_grad = []
        for observable in observables:

            paulis = [str(p[::-1]) for p in observable._pauli_list]
            coeff = list(np.real_if_close([c for c in observable.coeffs]))
            operator_real = Observable(self.num_qubits)
            operator_imag = Observable(self.num_qubits)
            operator_param = []
            operator_param_func = []
            operator_param_func_grad = []

            num_real = 0
            num_imag = 0
            num_param = 0
            for c, p in zip(coeff, paulis):
                string = ""
                for i, p_ in enumerate(p):
                    #if p_ != "I":
                    string += p_ + " " + str(i) + " "

                if isinstance(c, ParameterVectorElement):
                    # Single parameter vector element
                    operator_param_func.append(lambdify(self._symbol_tuple, sympify(c._symbol_expr)))
                    operator_param_func_grad.append([lambda x: 1.0])
                    self._free_parameters.add(c)
                    op = Observable(self.num_qubits)
                    op.add_operator(1.0, string)
                    operator_param.append(op)
                    num_param += 1

                elif isinstance(c, ParameterExpression):
                    # Parameter is in a expression (equation)
                    operator_param_func.append(lambdify(self._symbol_tuple, sympify(c._symbol_expr)))
                    func_grad_list_element = []
                    for param_element in c._parameter_symbols.keys():
                        self._free_parameters.add(param_element)
                        # information about the gradient of the parameter expression
                        # the 1j fixes a bug in qiskit
                        param_grad = -1j * ((1j * c).gradient(param_element))
                        if isinstance(param_grad, complex):
                            if param_grad.imag == 0:
                                param_grad = param_grad.real
                        if isinstance(param_grad, float) or isinstance(param_grad, complex):
                            # create a call by value labmda function
                            func_grad_list_element.append(lambda *arg, param_grad=param_grad: param_grad)
                        else:
                            func_grad_list_element.append(lambdify(self._symbol_tuple, sympify(param_grad._symbol_expr)))
                    operator_param_func_grad.append(func_grad_list_element)
                    op = Observable(self.num_qubits)
                    op.add_operator(1.0, string)
                    operator_param.append(op)
                    num_param += 1

                else:

                    operator_param = []
                    if np.abs(np.imag(c)) > 1e-12:
                        operator_imag.add_operator(np.imag(c), string)
                        num_imag += 1
                    if np.abs(np.real(c)) > 1e-12:
                        operator_real.add_operator(np.real(c), string)
                        num_real += 1

            if num_real == 0:
                operator_real = 0.0
            if num_imag == 0:
                operator_imag = 0.0

            self._operators_real.append(operator_real)
            self._operators_imag.append(operator_imag)
            self._operators_param.append(operator_param)
            self._operators_param_func.append(operator_param_func)
            self._operators_param_func_grad.append(operator_param_func_grad)


    def get_circuit_func(self):
        """Returns the Qulacs circuit function for the circuit."""

        def qulacs_circuit(*args):

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum(
                [list(args[i]) for i in range(len(self._qualcs_gates_parameters))], []
            )

            # Build the Qulacs circuit and evaluate the parametric terms
            circuit = ParametricQuantumCircuit(self.num_qubits)
            for i in range(len(self._operation_list)):
                if self._func_list[i] is None:
                    self._operation_list[i](circuit,*self._qubit_list[i])
                elif isinstance(self._func_list[i], float):
                    self._operation_list[i](circuit,self._func_list[i],*self._qubit_list[i])
                else:
                    value = self._func_list[i](*circ_param_list)
                    self._operation_list[i](circuit,value,*self._qubit_list[i])

            return circuit

        return qulacs_circuit

    def get_gradient_outer_jacobian(
        self,
        #parameters: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None,
    ):
        """Returns the outer jacobian needed for the chain rule in circuit derivatives.

        Qulacs does not support multiple parameters and parameter expressions,
        so we need to calculate a transformation which also includes the gradient of the
        parameter expression.
        """

        # # index of parameters to calculate the gradient for (all if parameters=None)
        # index_list = list(range(self.num_parameters))
        # if parameters is not None:
        #     if not isinstance(parameters, list):
        #         parameters = [parameters]
        #     index_list = [p.index for p in parameters]

        # enumerate the included parameter indices
        index_list = list(range(len(self._free_parameters)))
        index_dict = {}
        for i, index in enumerate(index_list):
            index_dict[index] = i

        def outer_jacobian(*args):

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum(
                [list(args[i]) for i in range(len(self._qualcs_gates_parameters))], []
            )

            # Count the number of parameter gates in the circuit
            num_param_in_circuit = sum(
                1
                for i in range(len(self._operation_list))
                if self._func_list[i] is not None and not isinstance(self._func_list[i], float)
            )

            outer_jacobian = np.zeros((num_param_in_circuit, len(index_list)))
            ioff = 0
            for i in range(len(self._operation_list)):
                if self._func_list[i] is not None:
                    outer_jacobian[ioff, index_dict[i]] = (
                        self._func_grad_list[i][0](*circ_param_list)
                    )
                    ioff += 1

            return outer_jacobian

        return outer_jacobian

def evaluate_circuit(circuit: QulacsCircuit, *args) -> np.ndarray:
    """
    Function to evaluate the Qulacs circuit with the given parameters.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        parameters (List[float]): List of parameters to evaluate the circuit

    Returns:
        np.ndarray: Result of the evaluation
    """

    # Collects the args values connected to the observable parameters
    obs_param_list = sum(
        [
            list(args[len(circuit._qualcs_gates_parameters) + i])
            for i in range(len(circuit._qualcs_obs_parameters))
        ],
        [],
    )

    circ = circuit.get_circuit_func()(*args[:len(circuit._qualcs_gates_parameters)])
    state = QuantumState(circuit.num_qubits)
    sim = QuantumCircuitSimulator(circ, state)
    sim.initialize_state(0)
    sim.simulate()

    real_values = np.array(
        [
            o if isinstance(o, float) else sim.get_expectation_value(o)
            for o in circuit._operators_real
        ]
    )
    imag_values = np.array(
        [
            o if isinstance(o, float) else sim.get_expectation_value(o)
            for o in circuit._operators_imag
        ]
    )
    param_obs_values = np.array(
        [
            [o if isinstance(o, float) else sim.get_expectation_value(o)
            for o in operator] for operator in circuit._operators_param
        ]
    )
    param_func_values =np.array([
        [0.0 if not callable(f) else f(*obs_param_list)
        for f in operator] for operator in circuit._operators_param_func
    ])

    # Compute the final parameter values by combining function and observable values
    param_values = np.array([
        np.dot(func_vals, obs_vals)
        for func_vals, obs_vals in zip(param_func_values, param_obs_values)
    ])

    values = np.real_if_close(real_values + 1j * imag_values + param_values)

    if not circuit.multiple_observables:
        return values[0]

    return values


def evaluate_circuit_gradient(circuit: QulacsCircuit, *args) -> np.ndarray:
    """
    Function to evaluate the Qulacs circuit with the given parameters.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        parameters (List[float]): List of parameters to evaluate the circuit

    Returns:
        np.ndarray: Result of the evaluation
    """

    # Collects the args values connected to the observable parameters
    obs_param_list = sum(
        [
            list(args[len(circuit._qualcs_gates_parameters) + i])
            for i in range(len(circuit._qualcs_obs_parameters))
        ],
        [],
    )

    qulacs_circuit = circuit.get_circuit_func()(*args[:len(circuit._qualcs_gates_parameters)])
    state = QuantumState(circuit.num_qubits)

    outer_jacobian = circuit.get_gradient_outer_jacobian()(*args[:len(circuit._qualcs_gates_parameters)])

    print("outer_jacobian",outer_jacobian)

    qulacs_circuit.update_quantum_state(state)

    qulacs_circuit2 = circuit.get_circuit_func()(*args[:len(circuit._qualcs_gates_parameters)])
    state = QuantumState(circuit.num_qubits)
    sim = QuantumCircuitSimulator(qulacs_circuit2, state)
    sim.initialize_state(0)
    sim.simulate()

    real_values = np.array(
        [
            (
                np.ones(outer_jacobian.shape[1]) * o
                if isinstance(o, float)
                else outer_jacobian.T @ np.array(qulacs_circuit.backprop(o))
            )
            for o in circuit._operators_real
        ]
    )
    imag_values = np.array(
        [
            (
                np.ones(outer_jacobian.shape[1]) * o
                if isinstance(o, float)
                else outer_jacobian.T @ np.array(qulacs_circuit.backprop(o))
            )
            for o in circuit._operators_imag
        ]
    )
    # param_obs_values = np.array(
    #     [
    #         [o if isinstance(o, float) else qulacs_circuit.backprop(o)
    #         for o in operator] for operator in circuit._operators_param
    #     ]
    # )
    
    # print("param_obs_values",param_obs_values)
    
    # param_func_values =np.array([
    #     [0.0 if not callable(f) else f(*obs_param_list)
    #     for f in operator] for operator in circuit._operators_param_func
    # ])

    # # Compute the final parameter values by combining function and observable values
    # param_values = np.array([
    #     np.dot(func_vals, obs_vals)
    #     for func_vals, obs_vals in zip(param_func_values, param_obs_values)
    # ])

    values = np.real_if_close(real_values + 1j * imag_values)
    
    #values = np.real_if_close(real_values + 1j * imag_values + param_values)

    if not circuit.multiple_observables:
        return values[0]

    return values




def evaluate_circuit_cc(circuit: QulacsCircuit, *args) -> np.ndarray:
    """
    Function to evaluate the Qulacs circuit with the given parameters.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        parameters (List[float]): List of parameters to evaluate the circuit

    Returns:
        np.ndarray: Result of the evaluation
    """

    # Collects the args values connected to the observable parameters
    obs_param_list = sum(
        [
            list(args[len(circuit._qualcs_gates_parameters) + i])
            for i in range(len(circuit._qualcs_obs_parameters))
        ],
        [],
    )

    circ = circuit.get_circuit_func()(*args[:len(circuit._qualcs_gates_parameters)])
    

    real_values = np.array(
        [
            o if isinstance(o, float) else CausalConeSimulator(circ, o).get_expectation_value()
            for o in circuit._operators_real
        ]
    )
    imag_values = np.array(
        [
            o if isinstance(o, float) else CausalConeSimulator(circ, o).get_expectation_value()
            for o in circuit._operators_imag
        ]
    )
    param_obs_values = np.array(
        [
            [o if isinstance(o, float) else CausalConeSimulator(circ, o).get_expectation_value()
            for o in operator] for operator in circuit._operators_param
        ]
    )
    param_func_values =np.array([
        [0.0 if not callable(f) else f(*obs_param_list)
        for f in operator] for operator in circuit._operators_param_func
    ])

    # Compute the final parameter values by combining function and observable values
    param_values = np.array([
        np.dot(func_vals, obs_vals)
        for func_vals, obs_vals in zip(param_func_values, param_obs_values)
    ])

    values = np.real_if_close(real_values + 1j * imag_values + param_values)

    if not circuit.multiple_observables:
        return values[0]

    return values
