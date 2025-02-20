import numpy as np
from typing import Union, List
from sympy import lambdify, sympify

from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.classicalregister import Clbit

from qulacs import QuantumCircuit, QuantumState, QuantumCircuitSimulator, CausalConeSimulator

from typing import List, Union, Iterable

from qiskit.compiler import transpile
from qiskit_aer import Aer

from qulacs import ParametricQuantumCircuit
from qulacs import Observable,GradCalculator, GeneralQuantumOperator, PauliOperator
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
        circuit (QiskitQuantumCircuit): Qiskit circuit to convert to Qulacs
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
        circuit: QiskitQuantumCircuit,
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
        self._used_parameters = []
        
        self._qualcs_gates_parameters = []
        self._symbol_tuple_circuit = tuple()
        

        self._rebuild_circuit_func = True
        self._circuit_func = None

        self._num_clbits = self._qiskit_circuit.num_clbits

        self._build_circuit_instructions(self._qiskit_circuit)

        self.new_operators = []
        self.new_operators_coeff = []
        self.new_operators_coeff_grad = []
        self.new_operators_used_parameters = []
        self.build_observable_instructions(self._qiskit_observable)

        #self._qulacs_circuit = self.build_qulacs_circuit()
        
        self._circuit_func_cache = {}
        self._outer_jacobi_circuit_cache = {}
        self._outer_jacobi_obs_cache = {}

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
        used_parameters = None

        # Change sign because of the way Qulacs defines the rotation gates
        angle = -angle

        if isinstance(angle, float):
            # Single float value
            func_list_element = angle
            func_grad_list_element = None
        elif isinstance(angle, ParameterVectorElement):
            # Single parameter vector element
            parameterized = True
            func_list_element = lambdify(self._symbol_tuple_circuit, sympify(angle._symbol_expr))
            func_grad_list_element = [lambda x: 1.0]
            self._free_parameters.add(angle)
            used_parameters = [angle]
            
        elif isinstance(angle, ParameterExpression):
            # Parameter is in a expression (equation)
            parameterized = True
            func_list_element = lambdify(self._symbol_tuple_circuit, sympify(angle._symbol_expr))
            func_grad_list_element = []
            used_parameters = []
            for param_element in angle._parameter_symbols.keys():
                self._free_parameters.add(param_element)
                used_parameters.append(param_element)
                # information about the gradient of the parameter expression
                param_grad = angle.gradient(param_element)
                if isinstance(param_grad, float):
                    # create a call by value labmda function
                    func_grad_list_element.append(lambda *arg, param_grad=param_grad: param_grad)
                else:
                    func_grad_list_element.append(lambdify(self._symbol_tuple_circuit, sympify(param_grad._symbol_expr)))

        return func_list_element, func_grad_list_element, used_parameters, parameterized

    def _add_single_qubit_gate(self, gate_name: str, qubits: Union[int, Iterable[int]]):
        """
        Adds a single qubit gate to the circuit.

        Args:
            gate_name (str): Name of the gate
            qubits (int or Iterable[int]): qubit indices
        """
        qubits = [qubits] if isinstance(qubits, int) else qubits
        for q in qubits:
            self._operation_list.append(gate_name)
            if q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} is out of range")
            self._qubit_list.append([q])
            self._func_list.append(None)
            self._func_grad_list.append(None)
            self._used_parameters.append([])

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
            self._operation_list.append(gate_name)
            self._qubit_list.append([control, target])
            self._func_list.append(None)
            self._func_grad_list.append(None)
            self._used_parameters.append([])

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
        func_list_element, func_grad_list_element, used_parameters, parameterized = (
            self._add_parameter_expression(angle)
        )

        qubits = [qubits] if isinstance(qubits, int) else qubits
        for q in qubits:
            if q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} is out of range")
            if parameterized:
                self._operation_list.append(gate_name)
            else:
                self._operation_list.append(gate_name)
            self._qubit_list.append([q])
            self._func_list.append(func_list_element)
            self._func_grad_list.append(func_grad_list_element)
            if used_parameters is None:
                self._used_parameters.append([])
            else:
                self._used_parameters.append(used_parameters)

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
        func_list_element, func_grad_list_element, used_parameters, parameterized = (
            self._add_parameter_expression(angle)
        )

        qubit1 = [qubit1] if isinstance(qubit1, int) else qubit1
        qubit2 = [qubit2] if isinstance(qubit2, int) else qubit2

        for control, target in zip(qubit1, qubit2):
            if control >= self.num_qubits or target >= self.num_qubits:
                raise ValueError(f"Qubit index is out of range")
            if parameterized:
                self._operation_list.append(gate_name)
            else:
                self._operation_list.append(gate_name)
            self._qubit_list.append([control, target])
            self._func_list.append(func_list_element)
            self._func_grad_list.append(func_grad_list_element)
            if used_parameters is None:
                self._used_parameters.append([])
            else:
                self._used_parameters.append(used_parameters)

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
        self._symbol_tuple_circuit = tuple()

        for param in circuit.parameters:
            if param.vector.name not in self._qualcs_gates_parameters:
                self._qualcs_gates_parameters.append(param.vector.name)

        self._symbol_tuple_circuit = tuple([sympify(p._symbol_expr) for p in circuit.parameters])

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

        self._symbol_tuple_obs = tuple()

        self._qualcs_obs_parameters = []


        for observable in observables:
            for param in observable.parameters:
                if param.vector.name not in self._qualcs_obs_parameters:
                    self._qualcs_obs_parameters.append(param.vector.name)

        def sort_parameters_after_index(parameter_vector):
            index_list = [p.index for p in parameter_vector]
            argsort_list = np.argsort(index_list)
            return [parameter_vector[i] for i in argsort_list]


        self._symbol_tuple_obs = tuple(
            sum(
                [
                    [sympify(p._symbol_expr) for p in sort_parameters_after_index(obs.parameters)]
                    for obs in observables
                ],
                [],
            )
        )

        # new version
        self.new_operators = []
        self.new_operators_coeff = []
        self.new_operators_coeff_grad = []
        self.new_operators_used_parameters = []
        for observable in observables:

            paulis = [str(p[::-1]) for p in observable._pauli_list]
            coeff = list(np.real_if_close([c for c in observable.coeffs]))

            new_operator = []
            new_operators_coeff = []
            new_operators_coeff_grad = []
            new_operators_used_parameters = []
            for c, p in zip(coeff, paulis):
                string = ""
                for i, p_ in enumerate(p):
                    #if p_ != "I":
                    string += p_ + " " + str(i) + " "

                new_operator.append(string)

                if isinstance(c, ParameterVectorElement):
                    # Single parameter vector element
                    new_operators_coeff.append(lambdify(self._symbol_tuple_obs, sympify(c._symbol_expr)))
                    new_operators_coeff_grad.append([lambda *arg: 1.0])
                    self._free_parameters.add(c)
                    new_operators_used_parameters.append([c])

                elif isinstance(c, ParameterExpression):
                    # Parameter is in a expression (equation)
                    new_operators_coeff.append(lambdify(self._symbol_tuple_obs, sympify(c._symbol_expr)))
                    func_grad_list_element = []
                    used_parameters_obs_element = []
                    for param_element in c._parameter_symbols.keys():
                        self._free_parameters.add(param_element)
                        used_parameters_obs_element.append(param_element)
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
                            func_grad_list_element.append(lambdify(self._symbol_tuple_obs, sympify(param_grad._symbol_expr)))
                    new_operators_coeff_grad.append(func_grad_list_element)
                    new_operators_used_parameters.append(used_parameters_obs_element)

                else:
                    new_operators_coeff.append(lambda *arg, c=c: c)
                    new_operators_coeff_grad.append([lambda *arg: 0.0])
                    new_operators_used_parameters.append([])

            self.new_operators.append(new_operator)
            self.new_operators_coeff.append(new_operators_coeff)
            self.new_operators_coeff_grad.append(new_operators_coeff_grad)
            self.new_operators_used_parameters.append(new_operators_used_parameters)

    def get_observable_func(self):
        """ Returns the Qulacs observable function for the observable depending on parameters."""

        def observable_func(*args):

            list_operators=[]
            for i,observable in enumerate(self.new_operators):
                operator = GeneralQuantumOperator(self.num_qubits)
                for j,op in enumerate(observable):

                    operator.add_operator(self.new_operators_coeff[i][j](*args),op)
                list_operators.append(operator)

            return list_operators

        return observable_func



    def get_circuit_func(self,gradient_param = None):
        """Returns the Qulacs circuit function for the circuit."""

        if isinstance(gradient_param, ParameterVectorElement):
            gradient_param = [gradient_param]
        gradient_param = list(gradient_param) if gradient_param is not None else []

        is_parameterized = len(gradient_param)
        parameterized_operations = [
            any(param in gradient_param for param in self._used_parameters[i]) for i, _ in enumerate(self._operation_list)
        ]

        cache_value = "no_gradient"
        if is_parameterized:
            cache_value = tuple(gradient_param)

        if cache_value in self._circuit_func_cache:
            return self._circuit_func_cache[cache_value]

        def qulacs_circuit(*args):

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum(
                [list(args[i]) for i in range(len(self._qualcs_gates_parameters))], []
            )

            if is_parameterized:
                circuit = ParametricQuantumCircuit(self.num_qubits)
            else:
                circuit = QuantumCircuit(self.num_qubits)

            # Build the Qulacs circuit and evaluate the parametric terms
            for i, op in enumerate(self._operation_list):
                if self._func_list[i] is None:
                    qiskit_qulacs_gate_dict[op](circuit,*self._qubit_list[i])
                elif isinstance(self._func_list[i], float):
                    qiskit_qulacs_gate_dict[op](circuit,self._func_list[i],*self._qubit_list[i])
                else:
                    value = self._func_list[i](*circ_param_list)
                    if parameterized_operations[i]:
                        qiskit_qulacs_param_gate_dict[op](circuit,value,*self._qubit_list[i])
                    else:
                        qiskit_qulacs_gate_dict[op](circuit,value,*self._qubit_list[i])

            return circuit

        self._circuit_func_cache[cache_value] = qulacs_circuit

        return qulacs_circuit

    def get_gradient_outer_jacobian(
        self,
        gradient_parameters: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None,
    ):
        """Returns the outer jacobian needed for the chain rule in circuit derivatives.

        Qulacs does not support multiple parameters and parameter expressions,
        so we need to calculate a transformation which also includes the gradient of the
        parameter expression.

        Args:
            gradient_parameters (Union[None, ParameterVectorElement, List[ParameterVectorElement]]): Parameters to calculate the gradient for
        """

        if isinstance(gradient_parameters, ParameterVectorElement):
            gradient_parameters = [gradient_parameters]
        gradient_parameters = list(gradient_parameters) if gradient_parameters is not None else []
        gradient_param_dict = {p:i for i,p in enumerate(gradient_parameters)}

        cache_value = "no_gradient"
        if len(gradient_parameters)>0:
            cache_value = tuple(gradient_parameters)

        if cache_value in self._outer_jacobi_circuit_cache:
            return self._outer_jacobi_circuit_cache[cache_value]

        def outer_jacobian(*args):

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum(
                [list(args[i]) for i in range(len(self._qualcs_gates_parameters))], []
            )

            relevant_operations = [
                i for i in range(len(self._operation_list))
                if any(param in gradient_parameters for param in self._used_parameters[i])
                ]

            outer_jacobian = np.zeros((len(relevant_operations), len(gradient_parameters)))

            for i,operation in enumerate(relevant_operations):
                for j,param in enumerate(self._used_parameters[operation]):
                    if param in gradient_parameters:
                        outer_jacobian[i, gradient_param_dict[param]] = (
                            self._func_grad_list[operation][j](*circ_param_list)
                        )

            return outer_jacobian

        self._outer_jacobi_circuit_cache[cache_value] = outer_jacobian

        return outer_jacobian

    def get_gradient_outer_jacobian_observables_new(
        self,
        gradient_parameters: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None,
    ):
        """Returns the outer jacobian needed for the chain rule in circuit derivatives.

        Qulacs does not support multiple parameters and parameter expressions,
        so we need to calculate a transformation which also includes the gradient of the
        parameter expression.

        Args:
            gradient_parameters (Union[None, ParameterVectorElement, List[ParameterVectorElement]]): Parameters to calculate the gradient for
        """

        if isinstance(gradient_parameters, ParameterVectorElement):
            gradient_parameters = [gradient_parameters]
        gradient_parameters = list(gradient_parameters) if gradient_parameters is not None else []
        gradient_param_dict = {p:i for i,p in enumerate(gradient_parameters)}

        # cache_value = "no_gradient"
        # if len(gradient_parameters)>0:
        #     cache_value = tuple(gradient_parameters)

        # if cache_value in self._outer_jacobi_obs_cache:
        #     return self._outer_jacobi_obs_cache[cache_value]

        def outer_jacobian(*args):

            # Collects the args values connected to the observable parameters
            obs_param_list = sum(
                [list(args[i]) for i in range(len(self._qualcs_obs_parameters))], []
            )

            outer_jacobians = []

            for iop,operator in enumerate(self.new_operators_coeff_grad):

                relevant_operations = [
                    i for i in range(len(operator))
                    if any(param in gradient_parameters for param in self.new_operators_used_parameters[iop][i])
                    ]

                outer_jacobian = np.zeros((len(relevant_operations), len(gradient_parameters)))
                for i,operation in enumerate(relevant_operations):
                    for j,param in enumerate(self.new_operators_used_parameters[iop][operation]):
                        if param in gradient_parameters:
                            outer_jacobian[i,gradient_param_dict[param]] = self.new_operators_coeff_grad[iop][operation][j](*obs_param_list)
                outer_jacobians.append(outer_jacobian)
            return outer_jacobians

        #self._outer_jacobi_obs_cache[cache_value] = outer_jacobian

        return outer_jacobian

    def get_operators_for_gradient(self,
                                    gradient_parameters: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None):

        """ Returns the Qulacs observable function for the observable depending on parameters."""

        if isinstance(gradient_parameters, ParameterVectorElement):
            gradient_parameters = [gradient_parameters]
        gradient_parameters = list(gradient_parameters) if gradient_parameters is not None else []

        def observable_func(*args):

            list_operators=[]
            for iop, observable in enumerate(self.new_operators):

                relevant_operations = [
                    i for i in range(len(observable))
                    if any(param in gradient_parameters for param in self.new_operators_used_parameters[iop][i])
                    ]

                list_paulis = []
                for op in relevant_operations:
                    list_paulis.append(PauliOperator(observable[op],1.0))
                list_operators.append(list_paulis)

            return list_operators

        return observable_func

def evaluate_circuit(circuit: QulacsCircuit, *args) -> np.ndarray:
    """
    Function to evaluate the Qulacs circuit with the given parameters.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        parameters (List[float]): List of parameters to evaluate the circuit

    Returns:
        np.ndarray: Result of the evaluation
    """

    # TODO: Check if this is desired functionality
    # Collects the args values connected to the observable parameters
    # obs_param_list = sum(
    #     [
    #         list(args[len(circuit._qualcs_gates_parameters) + i])
    #         for i in range(len(circuit._qualcs_obs_parameters))
    #     ],
    #     [],
    # )
    if circuit._qualcs_obs_parameters:
        obs_param_list = list(args[-1])
    else:
        obs_param_list = []

    circ = circuit.get_circuit_func()(*[args[i] for i, arg in enumerate(["param", "x"]) if arg in circuit._qualcs_gates_parameters])
    state = QuantumState(circuit.num_qubits)
    circ.update_quantum_state(state)

    operators = circuit.get_observable_func()(*obs_param_list)

    param_values = np.array([
                o.get_expectation_value(state)
                for o in operators
    ])

    values = np.real_if_close(param_values)

    if not circuit.multiple_observables:
        return values[0]

    return values

def evaluate_circuit_statevec(circuit: QulacsCircuit, *args) -> np.ndarray:
    """
    Function to evaluate the statevector of the Qulacs circuit with the given parameters.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        parameters (List[float]): List of parameters to evaluate the circuit

    Returns:
        np.ndarray: Statevector solution of the circuit
    """

    # TODO: Check if this is desired functionality
    # Collects the args values connected to the observable parameters
    circ = circuit.get_circuit_func()(*[args[i] for i, arg in enumerate(["param", "x"]) if arg in circuit._qualcs_gates_parameters])
    state = QuantumState(circuit.num_qubits)
    circ.update_quantum_state(state)

    return state.get_vector()

def evaluate_circuit_gradient(circuit: QulacsCircuit,
                              parameters: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None,
                              *args) -> np.ndarray:
    """
    Function to evaluate the Qulacs circuit with the given parameters.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        parameters (List[float]): List of parameters to evaluate the circuit

    Returns:
        np.ndarray: Result of the evaluation
    """

    # TODO: Check if this is desired functionality
    # Collects the args values connected to the observable parameters
    # obs_param_list = sum(
    #     [
    #         list(args[len(circuit._qualcs_gates_parameters) + i])
    #         for i in range(len(circuit._qualcs_obs_parameters))
    #     ],
    #     [],
    # )
    if circuit._qualcs_obs_parameters:
        obs_param_list = list(args[-1])
    else:
        obs_param_list = []


    qulacs_circuit = circuit.get_circuit_func(parameters)(*[args[i] for i, arg in enumerate(["param", "x"]) if arg in circuit._qualcs_gates_parameters])
    outer_jacobian = circuit.get_gradient_outer_jacobian(parameters)(*[args[i] for i, arg in enumerate(["param", "x"]) if arg in circuit._qualcs_gates_parameters])
    operators = circuit.get_observable_func()(*obs_param_list)

    is_parameterized = len(sum([list(param) for param in parameters if param],[]))

    if is_parameterized:
        param_values = np.array([
                    outer_jacobian.T @ np.array(qulacs_circuit.backprop(o))
                    for o in operators
        ])
    else:
        param_values = np.array([[]])

    values = np.real_if_close(param_values)

    if not circuit.multiple_observables:
        return values[0]

    return values



def evaluate_operator_gradient(circuit: QulacsCircuit,
                              parameters: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None,
                              *args) -> np.ndarray:
    """
    Function to evaluate the Qulacs circuit with the given parameters.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        parameters (List[float]): List of parameters to evaluate the circuit

    Returns:
        np.ndarray: Result of the evaluation
    """

    # TODO: Check if this is desired functionality
    # Collects the args values connected to the observable parameters
    # obs_param_list = sum(
    #     [
    #         list(args[len(circuit._qualcs_gates_parameters) + i])
    #         for i in range(len(circuit._qualcs_obs_parameters))
    #     ],
    #     [],
    # )
    if circuit._qualcs_obs_parameters:
        obs_param_list = list(args[-1])
    else:
        obs_param_list = []

    obs_param_list = [obs_param_list]
    outer_jacobian_new = circuit.get_gradient_outer_jacobian_observables_new(parameters)(*obs_param_list)

    circ = circuit.get_circuit_func()(*[args[i] for i, arg in enumerate(["param", "x"]) if arg in circuit._qualcs_gates_parameters])
    state = QuantumState(circuit.num_qubits)
    circ.update_quantum_state(state)
    operators = circuit.get_operators_for_gradient(parameters)()

    param_obs_values = [
        outer_jacobian_new[i].T @ np.array(
            [o if isinstance(o, float) else o.get_expectation_value(state) for o in operator]
        )
        for i, operator in enumerate(operators)
    ]

    values = np.real_if_close(param_obs_values)

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


    # TODO: Check if this is desired functionality
    #  Collects the args values connected to the observable parameters
    # obs_param_list = sum(
    #     [
    #         list(args[len(circuit._qualcs_gates_parameters) + i])
    #         for i in range(len(circuit._qualcs_obs_parameters))
    #     ],
    #     [],
    # )
    if circuit._qualcs_obs_parameters:
        obs_param_list = list(args[-1])
    else:
        obs_param_list = []

    circ = circuit.get_circuit_func()(*[args[i] for i, arg in enumerate(["param", "x"]) if arg in circuit._qualcs_gates_parameters])

    operators = circuit.get_observable_func()(*obs_param_list)

    param_values = np.array([
                CausalConeSimulator(circ, o).get_expectation_value()
                for o in operators
    ])

    values = np.real_if_close(param_values)

    if not circuit.multiple_observables:
        return values[0]

    return values
