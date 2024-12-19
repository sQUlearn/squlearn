import numpy as np
from typing import Union, List
from sympy import lambdify, sympify

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.classicalregister import Clbit

from qiskit.compiler import transpile
from qiskit_aer import Aer

import pennylane as qml
import pennylane.numpy as pnp
import pennylane.pauli as pauli
from pennylane.operation import Observable as PennyLaneObservable

from .pennylane_gates import qiskit_pennylane_gate_dict
from ..executor import Executor
from ..decompose_to_std import decompose_to_std


def _get_sympy_interface():
    """
    Returns the sympy interface that is used in the parameter conversion.

    Necessary for the correct conversion of sympy expressions in Qiskit to
    python functions in PennyLane.

    Returns:
        Tuple of sympy printer and sympy modules
    """
    # SymPy printer for pennylane numpy implementation has to be set manually,
    # otherwise math functions are used in lambdify instead of pennylane.numpy functions
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

    #     # SymPy printer for pennylane numpy implementation has to be set manually,
    #     # otherwise math functions are used in lambdify instead of pennylane.numpy functions
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


class PennyLaneCircuit:
    """
    Class for converting a Qiskit circuit to a PennyLane circuit.

    Args:
        circuit (QuantumCircuit): Qiskit circuit to convert to PennyLane
        observable (Union[None, SparsePauliOp, List[SparsePauliOp], str]): Observable to be measured
                                                                           Can be also a string like ``"probs"`` or ``"state"``
        executor (Executor): Executor object to handle the PennyLane circuit. Has to be initialized with a PennyLane device.

    Attributes:
    -----------

    Attributes:
        pennylane_circuit (qml.qnode): PennyLane circuit that can be called with parameters
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
            PennyLaneObservable,
            List[PennyLaneObservable],
        ] = None,
        executor: Executor = None,
    ) -> None:

        self._executor = executor
        if self._executor is None:
            self._executor = Executor("pennylane")

        # Transpile circuit to supported basis gates and expand blocks automatically
        self._qiskit_circuit = transpile(
            decompose_to_std(circuit),
            basis_gates=qiskit_pennylane_gate_dict.keys(),
            optimization_level=0,
        )

        self._qiskit_observable = observable
        self._num_qubits = self._qiskit_circuit.num_qubits

        # Build circuit instructions for the pennylane circuit from the qiskit circuit
        (
            self._pennylane_gates,
            self._pennylane_gates_param_function,
            self._pennylane_gates_wires,
            self._pennylane_conditions,
            self._pennylane_gates_parameters,
            self._pennylane_gates_parameters_dimensions,
        ) = self.build_circuit_instructions(self._qiskit_circuit)

        self._is_qiskit_observable = False
        if isinstance(observable, SparsePauliOp):
            self._is_qiskit_observable = True
        if isinstance(observable, list):
            if all([isinstance(obs, SparsePauliOp) for obs in observable]):
                self._is_qiskit_observable = True

        # Build circuit instructions for the pennylane observable from the qiskit circuit
        if self._is_qiskit_observable:
            (
                self._pennylane_obs_param_function,
                self._pennylane_words,
                self._pennylane_obs_parameters,
                self._pennylane_obs_parameters_dimensions,
            ) = self.build_observable_instructions(observable)
        else:
            self._pennylane_obs_param_function = []
            self._pennylane_words = []
            self._pennylane_obs_parameters = []

        self._num_clbits = self._qiskit_circuit.num_clbits

        self._pennylane_circuit = self.build_pennylane_circuit()

    @property
    def pennylane_circuit(self) -> callable:
        """PennyLane circuit that can be called with parameters"""
        return self._pennylane_circuit

    @property
    def circuit_parameter_names(self) -> list:
        """List of circuit parameter names"""
        return self._pennylane_gates_parameters

    @property
    def observable_parameter_names(self) -> list:
        """List of observable parameter names"""
        return self._pennylane_obs_parameters

    @property
    def circuit_parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each circuit parameter"""
        return self._pennylane_gates_parameters_dimensions

    @property
    def observable_parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each observable parameter"""
        return self._pennylane_obs_parameters_dimensions

    @property
    def circuit_arguments(self) -> list:
        """List of all circuit and observable parameters names"""
        return self._pennylane_gates_parameters + self._pennylane_obs_parameters

    @property
    def hash(self) -> str:
        """Hashable object of the circuit and observable for caching"""
        return str(self._qiskit_circuit) + str(self._qiskit_observable)

    def draw(self, engine: str = "pennylane", **kwargs):
        """Draw the circuit with the specified engine

        Args:
            engine (str): Engine to draw the circuit. Can be either ``"pennylane"`` or ``"qiskit"``
            **kwargs: Additional arguments for the drawing engine (only for qiskit)

        Returns:
            matplotlib Figure object of the circuit visualization
        """
        if engine == "pennylane":
            args = []
            for name in self.circuit_parameter_names:
                args.append(np.random.rand(self.circuit_parameter_dimensions[name]))
            for name in self.observable_parameter_names:
                args.append(np.random.rand(self.observable_parameter_dimensions[name]))
            args = tuple(args)
            return qml.draw_mpl(self._pennylane_circuit)(*args)
        elif engine == "qiskit":
            return self._qiskit_circuit.draw(**kwargs)
        else:
            raise NotImplementedError("Circuit engine not implemented")

    def get_pennylane_circuit(self) -> callable:
        """Builds and returns the PennyLane circuit as callable function"""
        self._pennylane_circuit = self.build_pennylane_circuit()
        return self._pennylane_circuit

    def __call__(self, *args, **kwargs):
        return self._pennylane_circuit(*args, **kwargs)

    def build_circuit_instructions(self, circuit: QuantumCircuit) -> tuple:
        """
        Function to build the instructions for the PennyLane circuit from the Qiskit circuit.

        This functions converts the Qiskit gates and parameter expressions to PennyLane compatible
        gates and functions.

        Args:
            circuit (QuantumCircuit): Qiskit circuit to convert to PennyLane

        Returns:
            Tuple with lists of PennyLane gates, PennyLane gate parameter functions,
            PennyLane gate wires, PennyLane gate parameters and PennyLane gate parameter dimensions
        """
        pennylane_gates = []
        pennylane_gates_param_function = []
        pennylane_gates_wires = []
        pennylane_conditions = []
        pennylane_gates_parameters = []
        pennylane_gates_parameters_dimensions = {}

        symbol_tuple = tuple([sympify(p._symbol_expr) for p in circuit.parameters])

        for param in circuit.parameters:
            if param.vector.name not in pennylane_gates_parameters:
                pennylane_gates_parameters.append(param.vector.name)
                pennylane_gates_parameters_dimensions[param.vector.name] = 1
            else:
                pennylane_gates_parameters_dimensions[param.vector.name] += 1

        printer, modules = _get_sympy_interface()

        for op in circuit.data:

            # catch conditions of the gate
            # only c_if is supported, the other cases have been caught before
            if op.operation.condition is None:
                # No condition (usually the case)
                pennylane_conditions.append(None)
            else:
                classical_bits = op.operation.condition[0]
                val = op.operation.condition[1]
                if isinstance(classical_bits, Clbit):
                    i = circuit.find_bit(classical_bits).index
                else:
                    i = [circuit.find_bit(b).index for b in classical_bits]
                # add indices of classical bits containing measured values
                # and value of the conditions (measurement equal to val)
                pennylane_conditions.append((i, val))

            param_tuple = None
            if len(op.operation.params) >= 1:
                param_tuple = ()
                for param in op.operation.params:
                    if isinstance(param, ParameterExpression):
                        if param._symbol_expr == None:
                            param = param._coeff
                        else:
                            symbol_expr = sympify(param._symbol_expr)
                            f = lambdify(
                                symbol_tuple, symbol_expr, modules=modules, printer=printer
                            )

                            param_tuple += (f,)
                    else:
                        param_tuple += (param,)

            pennylane_gates_param_function.append(param_tuple)

            if op.operation.name == "measure":
                # Capture special case of measurement, that is stored in classical bits
                # In the pennylane implementation, classical bits are introduced as an array
                wires = [
                    circuit.find_bit(op.qubits[i]).index for i in range(op.operation.num_qubits)
                ]
                clbits = [
                    circuit.find_bit(op.clbits[i]).index for i in range(op.operation.num_clbits)
                ]
                pennylane_gates.append(("measure", clbits))
                pennylane_gates_wires.append(wires)
            else:
                # All other gates
                if op.operation.name not in qiskit_pennylane_gate_dict:
                    raise NotImplementedError(
                        f"Gate {op.operation.name} is unfortunatly not supported in sQUlearn's PennyLane backend."
                    )

                pennylane_gates.append(qiskit_pennylane_gate_dict[op.operation.name])
                wires = [
                    circuit.find_bit(op.qubits[i]).index for i in range(op.operation.num_qubits)
                ]
                pennylane_gates_wires.append(wires)

        return (
            pennylane_gates,
            pennylane_gates_param_function,
            pennylane_gates_wires,
            pennylane_conditions,
            pennylane_gates_parameters,
            pennylane_gates_parameters_dimensions,
        )

    def build_observable_instructions(self, observable: Union[List[SparsePauliOp], SparsePauliOp]):
        """
        Function to build the instructions for the PennyLane observable from the Qiskit observable.

        This functions converts the Qiskit SparsePauli and parameter expressions to PennyLane
        compatible Pauli words and functions.

        Args:
            observable (Union[List[SparsePauliOp], SparsePauliOp]): Qiskit observable to convert
                                                                    to PennyLane

        Returns:
            Tuple with lists of PennyLane observable parameter functions, PennyLane Pauli words,
            PennyLane observable parameters and PennyLane observable parameter dimensions
        """
        if observable == None:
            return None, None, None

        pennylane_obs_param_function = []
        pennylane_obs_parameters = []
        pennylane_words = []

        islist = True
        if not isinstance(observable, list):
            islist = False
            observable = [observable]

        def sort_parameters_after_index(parameter_vector):
            index_list = [p.index for p in parameter_vector]
            argsort_list = np.argsort(index_list)
            return [parameter_vector[i] for i in argsort_list]

        printer, modules = _get_sympy_interface()

        # Get names of all parameters in all observables
        pennylane_obs_parameters = []
        pennylane_obs_parameters_dimensions = {}
        for obs in observable:
            for param in obs.parameters:
                if param.vector.name not in pennylane_obs_parameters:
                    pennylane_obs_parameters.append(param.vector.name)
                    pennylane_obs_parameters_dimensions[param.vector.name] = 1
                else:
                    pennylane_obs_parameters_dimensions[param.vector.name] += 1

        # Handle observable parameter expressions and convert them to compatible python functions

        symbol_tuple = tuple(
            sum(
                [
                    [sympify(p._symbol_expr) for p in sort_parameters_after_index(obs.parameters)]
                    for obs in observable
                ],
                [],
            )
        )

        pennylane_obs_param_function = []
        for obs in observable:
            pennylane_obs_param_function_ = []
            for coeff in obs.coeffs:
                if isinstance(coeff, ParameterExpression):
                    if coeff._symbol_expr == None:
                        coeff = coeff._coeff
                        if isinstance(coeff, np.complex128) or isinstance(coeff, np.complex64):
                            if np.imag(coeff) != 0:
                                raise ValueError(
                                    "Imaginary part of observable coefficient is not supported"
                                )
                            coeff = float(np.real(coeff))
                        else:
                            coeff = float(coeff)
                    else:
                        symbol_expr = sympify(coeff._symbol_expr)
                        f = lambdify(symbol_tuple, symbol_expr, modules=modules, printer=printer)
                        pennylane_obs_param_function_.append(f)
                else:
                    if isinstance(coeff, np.complex128) or isinstance(coeff, np.complex64):
                        if np.imag(coeff) != 0:
                            raise ValueError(
                                "Imaginary part of observable coefficient is not supported"
                            )
                        coeff = float(np.real(coeff))
                    else:
                        coeff = float(coeff)
                    pennylane_obs_param_function_.append(coeff)
            pennylane_obs_param_function.append(pennylane_obs_param_function_)

        # Convert Pauli strings into PennyLane Pauli words
        pennylane_words = []
        for obs in observable:
            pennylane_words.append(
                [pauli.string_to_pauli_word(str(p[::-1])) for p in obs._pauli_list]
            )

        if islist:
            return (
                pennylane_obs_param_function,
                pennylane_words,
                pennylane_obs_parameters,
                pennylane_obs_parameters_dimensions,
            )
        else:
            return (
                pennylane_obs_param_function[0],
                pennylane_words[0],
                pennylane_obs_parameters,
                pennylane_obs_parameters_dimensions,
            )

    def build_pennylane_circuit(self, max_diff: Union[int, None] = None):
        """
        Function to build the PennyLane circuit from the Qiskit circuit and observable.

        The functions returns a callable PennyLane circuit that can be called with parameters.
        The PennyLane circuit is built from the instructions previously generated from the Qiskit
        circuit and observable.

        Returns:
            Callable PennyLane circuit
        """

        if max_diff is None:
            max_diff = 1

        @qml.qnode(self._executor.backend, diff_method="best", max_diff=max_diff)
        def pennylane_circuit(*args):
            """PennyLane circuit that can be called with parameters"""

            measurements = [0] * self._num_clbits

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum(
                [list(args[i]) for i in range(len(self._pennylane_gates_parameters))], []
            )

            # Collects the args values connected to the observable parameters
            obs_param_list = sum(
                [
                    list(args[len(self._pennylane_gates_parameters) + i])
                    for i in range(len(self._pennylane_obs_parameters))
                ],
                [],
            )

            # Loop through all penny lane gates
            for i, op in enumerate(self._pennylane_gates):

                if isinstance(op, tuple):
                    # Special case for measurement
                    # add measurement to the circuit and store the result in the measurements array
                    if op[0] == "measure":
                        for j, wire in enumerate(self._pennylane_gates_wires[i]):
                            measurements[op[1][j]] = qml.measure(wire)
                else:
                    # Evaluate the (non-linear) parameter expression of the gate
                    if self._pennylane_gates_param_function[i] != None:
                        evaluated_param = tuple(
                            [
                                func(*circ_param_list) if callable(func) else func
                                for func in self._pennylane_gates_param_function[i]
                            ]
                        )
                    else:
                        evaluated_param = None

                    # Treat c_if conditions of the gate (if present)
                    if self._pennylane_conditions[i] != None:
                        # Calculate the value of the classical bit(s) involved in the condition
                        if isinstance(self._pennylane_conditions[i][0], list):
                            # conditions involving multiple classical bits -> convert to integer
                            val = 0
                            for j in range(len(self._pennylane_conditions[i][0])):
                                val += 2**j * measurements[self._pennylane_conditions[i][0][j]]
                        else:
                            val = measurements[self._pennylane_conditions[i][0]]

                        if evaluated_param is not None:
                            # The case that the gate has parameters
                            if isinstance(val, int):
                                # Conditional values are already integers
                                if val == self._pennylane_conditions[i][1]:
                                    op(*evaluated_param, wires=self._pennylane_gates_wires[i])
                            else:
                                # Otherwise, pennylane condition
                                qml.cond(val == self._pennylane_conditions[i][1], op)(
                                    *evaluated_param, wires=self._pennylane_gates_wires[i]
                                )
                        else:
                            # The case that the gate has no parameters
                            if isinstance(val, int):
                                # Conditional values are already integers
                                if val == self._pennylane_conditions[i][1]:
                                    op(wires=self._pennylane_gates_wires[i])
                            else:
                                # Otherwise, pennylane condition
                                qml.cond(val == self._pennylane_conditions[i][1], op)(
                                    wires=self._pennylane_gates_wires[i]
                                )
                    else:
                        if evaluated_param is not None:
                            op(*evaluated_param, wires=self._pennylane_gates_wires[i])
                        else:
                            op(wires=self._pennylane_gates_wires[i])

            # Defines the output of the circuit
            if self._qiskit_observable == None:
                return qml.probs(wires=range(self._num_qubits))
            elif self._qiskit_observable == "probs":
                return qml.probs(wires=range(self._num_qubits))
            elif self._qiskit_observable == "state":
                return qml.state()
            elif self._is_qiskit_observable:
                if isinstance(self._qiskit_observable, list):
                    expval_list = []
                    for i, obs in enumerate(self._pennylane_words):
                        if len(obs_param_list) > 0:
                            coeff_list = []
                            for coeff in self._pennylane_obs_param_function[i]:
                                if callable(coeff):
                                    evaluated_param = coeff(*obs_param_list)
                                    coeff_list.append(evaluated_param)
                                else:
                                    coeff_list.append(coeff)
                            expval_list.append(qml.expval(qml.Hamiltonian(coeff_list, obs)))
                        else:
                            # In case no parameters are present in the observable
                            # Calculate the expectation value of sum of the observables
                            # since this is more compatible with hardware backends
                            if len(self._pennylane_words[i]) == 0:
                                expval_list.append(0.0)
                            else:
                                expval_list.append(
                                    qml.expval(sum([obs for obs in self._pennylane_words[i]]))
                                )
                    return pnp.stack(tuple(expval_list))
                else:
                    if len(obs_param_list) > 0:
                        coeff_list = []
                        for coeff in self._pennylane_obs_param_function:
                            if callable(coeff):
                                evaluated_param = coeff(*obs_param_list)
                                coeff_list.append(evaluated_param)
                            else:
                                coeff_list.append(coeff)
                        return qml.expval(qml.Hamiltonian(coeff_list, self._pennylane_words))
                    else:
                        # In case no parameters are present in the observable
                        # Calculate the expectation value of sum of the observables
                        # since this is more compatible with hardware backends
                        if len(self._pennylane_words) == 0:
                            return 0.0
                        else:
                            return qml.expval(sum([obs for obs in self._pennylane_words]))
            else:
                if isinstance(self._qiskit_observable, list):
                    return pnp.stack(tuple([qml.expval(obs) for obs in self._qiskit_observable]))
                else:
                    return qml.expval(self._qiskit_observable)

        return pennylane_circuit
