import numpy as np
from typing import Union, List
from sympy import lambdify, sympify

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.classicalregister import Clbit


from typing import List, Union, Iterable

from qiskit.compiler import transpile
from qiskit_aer import Aer

from qulacs import ParametricQuantumCircuit

#import qulacs as qml
#import qulacs.numpy as pnp
#import qulacs.pauli as pauli
#from qulacs.operation import Observable as QulacsObservable

from .qulacs_gates import qiskit_qulacs_gate_dict
from ..executor import Executor
from ..decompose_to_std import decompose_to_std


def _get_sympy_interface():
    """
    Returns the sympy interface that is used in the parameter conversion.

    Necessary for the correct conversion of sympy expressions in Qiskit to
    python functions in Qulacs.

    Returns:
        Tuple of sympy printer and sympy modules
    """
    # SymPy printer for qulacs numpy implementation has to be set manually,
    # otherwise math functions are used in lambdify instead of qulacs.numpy functions
    from sympy.printing.numpy import NumPyPrinter as Printer

    user_functions = {}
    printer = Printer(
        {
            "fully_qualified_modules": False,
            "inline": True,
            "allow_unknown_functions": True,
            "user_functions": user_functions,
        }
    )
    # Use Pennylane numpy for sympy lambdify
    modules = pnp

    # The functions down below can be used to switch between different gradient engines
    # as tensorflow, jax and torch. However, this is not supported and implemented yet.

    #     # SymPy printer for qulacs numpy implementation has to be set manually,
    #     # otherwise math functions are used in lambdify instead of qulacs.numpy functions
    #     from sympy.printing.tensorflow import TensorflowPrinter as Printer  # type: ignore

    #     user_functions = {}
    #     printer = Printer(
    #         {
    #             "fully_qualified_modules": False,
    #             "inline": True,
    #             "allow_unknown_functions": True,
    #             "user_functions": user_functions,
    #         }
    #     )  #
    #     modules = tf

    # elif self._gradient_engine == "jax":
    #     from sympy.printing.numpy import JaxPrinter as Printer  # type: ignore

    #     user_functions = {}
    #     printer = Printer(
    #         {
    #             "fully_qualified_modules": False,
    #             "inline": True,
    #             "allow_unknown_functions": True,
    #             "user_functions": user_functions,
    #         }
    #     )  #
    #     modules = jnp
    # elif self._gradient_engine == "torch" or self._gradient_engine == "pytorch":
    #     from sympy.printing.pycode import PythonCodePrinter as Printer  # type: ignore

    #     user_functions = {}
    #     printer = Printer(
    #         {
    #             "fully_qualified_modules": False,
    #             "inline": True,
    #             "allow_unknown_functions": True,
    #             "user_functions": user_functions,
    #         }
    #     )  #
    #     modules = torch

    # else:
    #     # tbd for jax and tensorflow
    #     printer = None
    #     modules = None

    return printer, modules


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
        self._param_list = []
        self._qubit_list = []
        self._func_list = []
        self._func_grad_list = []
        self._free_parameters = set()

        self._rebuild_circuit_func = True
        self._circuit_func = None

        self._num_clbits = self._qiskit_circuit.num_clbits

        self._build_circuit_instructions(self._qiskit_circuit)

        #self._qulacs_circuit = self.build_qulacs_circuit()

    @property
    def num_qubits(self) -> int:
        """Number of qubits of the circuit"""
        return self._num_qubits

    @property
    def qulacs_circuit(self) -> callable:
        """Qulacs circuit that can be called with parameters"""
        return self._qulacs_circuit

    # @property
    # def circuit_parameter_names(self) -> list:
    #     """List of circuit parameter names"""
    #     return self._qulacs_gates_parameters

    # @property
    # def observable_parameter_names(self) -> list:
    #     """List of observable parameter names"""
    #     return self._qulacs_obs_parameters

    # @property
    # def circuit_parameter_dimensions(self) -> dict:
    #     """Dictionary with the dimension of each circuit parameter"""
    #     return self._qulacs_gates_parameters_dimensions

    # @property
    # def observable_parameter_dimensions(self) -> dict:
    #     """Dictionary with the dimension of each observable parameter"""
    #     return self._qulacs_obs_parameters_dimensions

    # @property
    # def circuit_arguments(self) -> list:
    #     """List of all circuit and observable parameters names"""
    #     return self._qulacs_gates_parameters + self._qulacs_obs_parameters

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
        param_list_element = angle
        func_list_element = None
        func_grad_list_element = None

        if isinstance(angle, ParameterVectorElement):
            # Single parameter vector element
            func_list_element = lambda x: x
            func_grad_list_element = lambda x: 1.0
            param_list_element = angle.index
            self._free_parameters.add(angle)
        elif isinstance(angle, ParameterExpression):
            # Parameter is in a expression (equation)
            if len(angle._parameter_symbols) > 1:
                raise ValueError("More then one free paramter per gate is not supported")

            param_element = list(angle._parameter_symbols.keys())[0]
            param_list_element = param_element.index
            self._free_parameters.add(param_element)
            symbol_tuple = (sympify(param_element._symbol_expr),)
            # Create a lambda function for the parameter expression
            func_list_element = lambdify(symbol_tuple, sympify(angle._symbol_expr))
            # Calculate gradient of the parameter expression
            param_grad = angle.gradient(param_element)
            if isinstance(param_grad, float):
                # create a call by value labmda function
                func_grad_list_element = lambda *arg, param_grad=param_grad: param_grad
            else:
                func_grad_list_element = lambdify(symbol_tuple, sympify(param_grad._symbol_expr))

        return param_list_element, func_list_element, func_grad_list_element

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
            self._param_list.append(None)
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
        qubit1 = [qubit1] * len(qubit2) if len(qubit1) == 1 else qubit1
        qubit2 = [qubit2] * len(qubit1) if len(qubit2) == 1 else qubit2

        for control, target in zip(qubit1, qubit2):
            if control >= self.num_qubits or target >= self.num_qubits:
                raise ValueError(f"Qubit index is out of range")
            self._operation_list.append(gate_name)
            self._qubit_list.append([control, target])
            self._param_list.append(None)
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
        param_list_element, func_list_element, func_grad_list_element = (
            self._add_parameter_expression(angle)
        )

        qubits = [qubits] if isinstance(qubits, int) else qubits
        for q in qubits:
            self._operation_list.append(gate_name)
            if q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} is out of range")
            self._qubit_list.append([q])
            self._param_list.append(param_list_element)
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
        param_list_element, func_list_element, func_grad_list_element = (
            self._add_parameter_expression(angle)
        )

        qubit1 = [qubit1] if isinstance(qubit1, int) else qubit1
        qubit2 = [qubit2] if isinstance(qubit2, int) else qubit2
        qubit1 = [qubit1] * len(qubit2) if len(qubit1) == 1 else qubit1
        qubit2 = [qubit2] * len(qubit1) if len(qubit2) == 1 else qubit2

        for control, target in zip(qubit1, qubit2):
            if control >= self.num_qubits or target >= self.num_qubits:
                raise ValueError(f"Qubit index is out of range")
            self._operation_list.append(gate_name)
            self._qubit_list.append([control, target])
            self._param_list.append(param_list_element)
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
      #  qulacs_gates = []
      #  qulacs_gates_param_function = []
      #  qulacs_gates_wires = []
      #  qulacs_conditions = []
      #  qulacs_gates_parameters = []
      #  qulacs_gates_parameters_dimensions = {}

        # symbol_tuple = tuple([sympify(p._symbol_expr) for p in circuit.parameters])

        # for param in circuit.parameters:
        #     if param.vector.name not in qulacs_gates_parameters:
        #         qulacs_gates_parameters.append(param.vector.name)
        #         qulacs_gates_parameters_dimensions[param.vector.name] = 1
        #     else:
        #         qulacs_gates_parameters_dimensions[param.vector.name] += 1

        # printer, modules = _get_sympy_interface()

        self._operation_list = []
        self._param_list = []
        self._qubit_list = []
        self._func_list = []
        self._func_grad_list = []
        self._free_parameters = set()

        for op in circuit.data:

            # catch conditions of the gate
            # only c_if is supported, the other cases have been caught before
            if op.operation.condition is not None or op.operation.name == "measure":
                raise NotImplementedError("Conditions are not supported in sQUlearn's Qulacs backend.")

            # param_tuple = None
            # if len(op.operation.params) >= 1:
            #     param_tuple = ()
            #     for param in op.operation.params:
            #         if isinstance(param, ParameterExpression):
            #             if param._symbol_expr == None:
            #                 param = param._coeff
            #             else:
            #                 symbol_expr = sympify(param._symbol_expr)
            #                 f = lambdify(
            #                     symbol_tuple, symbol_expr, modules=modules, printer=printer
            #                 )

            #                 param_tuple += (f,)
            #         else:
            #             param_tuple += (param,)

            # qulacs_gates_param_function.append(param_tuple)

            # if op.operation.name == "measure":
            #     # Capture special case of measurement, that is stored in classical bits
            #     # In the qulacs implementation, classical bits are introduced as an array
            #     wires = [
            #         circuit.find_bit(op.qubits[i]).index for i in range(op.operation.num_qubits)
            #     ]
            #     clbits = [
            #         circuit.find_bit(op.clbits[i]).index for i in range(op.operation.num_clbits)
            #     ]
            #     qulacs_gates.append(("measure", clbits))
            #     qulacs_gates_wires.append(wires)
            # else:
                # All other gates
            if op.operation.name not in qiskit_qulacs_gate_dict:
                raise NotImplementedError(
                    f"Gate {op.operation.name} is unfortunatly not supported in sQUlearn's Qulacs backend."
                )

            paramterized_gate = len(op.operation.params) >= 1

            # check for two qubit gates
            single_qubit_date = len(op.qubits) == 1

            wires = [
                circuit.find_bit(op.qubits[i]).index for i in range(op.operation.num_qubits)
            ]

            if single_qubit_date:
                if not paramterized_gate:
                    self._add_single_qubit_gate(qiskit_qulacs_gate_dict[op.operation.name], wires)
                else:
                    print("len(op.operation.params)",len(op.operation.params))
                    self._add_parameterized_single_qubit_gate(qiskit_qulacs_gate_dict[op.operation.name], wires, op.operation.params[0])
            else:
                if len(op.qubits)>2:
                    raise NotImplementedError("Only two qubit gates are supported in sQUlearn's Qulacs backend.")
                if not paramterized_gate:
                    self._add_parameterized_two_qubit_gate(qiskit_qulacs_gate_dict[op.operation.name], wires[0], wires[1])
                else:
                    print("len(op.operation.params)",len(op.operation.params))
                    self._add_parameterized_two_qubit_gate(qiskit_qulacs_gate_dict[op.operation.name], wires[0], wires[1], op.operation.params[0])

    # def build_observable_instructions(self, observable: Union[List[SparsePauliOp], SparsePauliOp]):
    #     """
    #     Function to build the instructions for the Qulacs observable from the Qiskit observable.

    #     This functions converts the Qiskit SparsePauli and parameter expressions to Qulacs
    #     compatible Pauli words and functions.

    #     Args:
    #         observable (Union[List[SparsePauliOp], SparsePauliOp]): Qiskit observable to convert
    #                                                                 to Qulacs

    #     Returns:
    #         Tuple with lists of Qulacs observable parameter functions, Qulacs Pauli words,
    #         Qulacs observable parameters and Qulacs observable parameter dimensions
    #     """
    #     if observable == None:
    #         return None, None, None

    #     qulacs_obs_param_function = []
    #     qulacs_obs_parameters = []
    #     qulacs_words = []

    #     islist = True
    #     if not isinstance(observable, list):
    #         islist = False
    #         observable = [observable]

    #     def sort_parameters_after_index(parameter_vector):
    #         index_list = [p.index for p in parameter_vector]
    #         argsort_list = np.argsort(index_list)
    #         return [parameter_vector[i] for i in argsort_list]

    #     printer, modules = _get_sympy_interface()

    #     # Get names of all parameters in all observables
    #     qulacs_obs_parameters = []
    #     qulacs_obs_parameters_dimensions = {}
    #     for obs in observable:
    #         for param in obs.parameters:
    #             if param.vector.name not in qulacs_obs_parameters:
    #                 qulacs_obs_parameters.append(param.vector.name)
    #                 qulacs_obs_parameters_dimensions[param.vector.name] = 1
    #             else:
    #                 qulacs_obs_parameters_dimensions[param.vector.name] += 1

    #     # Handle observable parameter expressions and convert them to compatible python functions

    #     symbol_tuple = tuple(
    #         sum(
    #             [
    #                 [sympify(p._symbol_expr) for p in sort_parameters_after_index(obs.parameters)]
    #                 for obs in observable
    #             ],
    #             [],
    #         )
    #     )

    #     qulacs_obs_param_function = []
    #     for obs in observable:
    #         qulacs_obs_param_function_ = []
    #         for coeff in obs.coeffs:
    #             if isinstance(coeff, ParameterExpression):
    #                 if coeff._symbol_expr == None:
    #                     coeff = coeff._coeff
    #                     if isinstance(coeff, np.complex128) or isinstance(coeff, np.complex64):
    #                         if np.imag(coeff) != 0:
    #                             raise ValueError(
    #                                 "Imaginary part of observable coefficient is not supported"
    #                             )
    #                         coeff = float(np.real(coeff))
    #                     else:
    #                         coeff = float(coeff)
    #                 else:
    #                     symbol_expr = sympify(coeff._symbol_expr)
    #                     f = lambdify(symbol_tuple, symbol_expr, modules=modules, printer=printer)
    #                     qulacs_obs_param_function_.append(f)
    #             else:
    #                 if isinstance(coeff, np.complex128) or isinstance(coeff, np.complex64):
    #                     if np.imag(coeff) != 0:
    #                         raise ValueError(
    #                             "Imaginary part of observable coefficient is not supported"
    #                         )
    #                     coeff = float(np.real(coeff))
    #                 else:
    #                     coeff = float(coeff)
    #                 qulacs_obs_param_function_.append(coeff)
    #         qulacs_obs_param_function.append(qulacs_obs_param_function_)

    #     # Convert Pauli strings into Qulacs Pauli words
    #     qulacs_words = []
    #     for obs in observable:
    #         qulacs_words.append(
    #             [pauli.string_to_pauli_word(str(p[::-1])) for p in obs._pauli_list]
    #         )

    #     if islist:
    #         return (
    #             qulacs_obs_param_function,
    #             qulacs_words,
    #             qulacs_obs_parameters,
    #             qulacs_obs_parameters_dimensions,
    #         )
    #     else:
    #         return (
    #             qulacs_obs_param_function[0],
    #             qulacs_words[0],
    #             qulacs_obs_parameters,
    #             qulacs_obs_parameters_dimensions,
    #         )

    def get_circuit_func(
        self,
        parameters: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None,
    ):
        """Returns the Qulacs circuit function for the circuit."""

        cache_function = True
        self._singleparameter_index_list = None
        if parameters is not None:
            cache_function = False
            self._rebuild_circuit_func = True
            if not isinstance(parameters, list):
                parameters = [parameters]
            self._singleparameter_index_list = [p.index for p in parameters]

        if self._rebuild_circuit_func:
            self._rebuild_circuit_func = not cache_function

            def qulacs_circuit(parameter):

                circuit = ParametricQuantumCircuit(self.num_qubits)
                for i in range(len(self._operation_list)):

                    self._operation_list[i](circuit,self._qubit_list[i][0])

                return circuit

            if cache_function:
                self._circuit_func = qulacs_circuit

        return self._circuit_func