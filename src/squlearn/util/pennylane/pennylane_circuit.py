import numpy as np
from typing import Union, List
import matplotlib.pyplot as plt
from sympy import lambdify
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.quantum_info import SparsePauliOp

import pennylane as qml
import pennylane.numpy as pnp
import pennylane.pauli as pauli

from .pennylane_device import PennyLaneDevice
from .pennylane_gates import qiskit_pennyland_gate_dict


class PennyLaneCircuit:

    def __init__(
        self,
        device: PennyLaneDevice,
        circuit: QuantumCircuit,
        observable: Union[None, SparsePauliOp, List[SparsePauliOp]] = None,
    ) -> None:

        self._device = device
        self._qiskit_circuit = circuit
        self._qiskit_observable = observable
        self._num_qubits = self._qiskit_circuit.num_qubits

        # Build circuit instructions for the pennylane circuit from the qiskit circuit
        # self._pennylane_gates = []
        # self._pennylane_gates_param_function = []
        # self._pennylane_gates_wires = []
        # self._pennylane_gates_parameters = []
        # self._build_circuit_instructions()
        (
            self._pennylane_gates,
            self._pennylane_gates_param_function,
            self._pennylane_gates_wires,
            self._pennylane_gates_parameters,
        ) = self.build_circuit_instructions(self._qiskit_circuit)
        print("self._pennylane_gates", self._pennylane_gates)
        print("self._pennylane_gates_param_function", self._pennylane_gates_param_function)
        print("self._pennylane_gates_wires", self._pennylane_gates_wires)
        print("self._pennylane_gates_parameters", self._pennylane_gates_parameters)

        # Build circuit instructions for the pennylane observable from the qiskit circuit

        if observable != None:
            # if isinstance(observable, list):
            #     self._pennylane_obs_param_function = []
            #     self._pennylane_words = []
            #     self._pennylane_obs_parameters = []
            #     for obs in observable:
            #         pennylane_obs_param_function, pennylane_words, pennylane_obs_parameters = self.build_observable_instructions(obs)
            #         self._pennylane_obs_param_function += pennylane_obs_param_function
            #         self._pennylane_words += pennylane_words
            #         self._pennylane_obs_parameters += pennylane_obs_parameters
            # else:
            (
                self._pennylane_obs_param_function,
                self._pennylane_words,
                self._pennylane_obs_parameters,
            ) = self.build_observable_instructions(observable)
            print("self._pennylane_obs_param_function", self._pennylane_obs_param_function)
            print("self._pennylane_words", self._pennylane_words)
            print("self._pennylane_obs_parameters", self._pennylane_obs_parameters)

        self._pennylane_circuit = self.build_pennylane_circuit()

    @property
    def pennylane_circuit(self):
        return self._pennylane_circuit

    @property
    def circuit_parameter_names(self) -> list:
        return self._pennylane_gates_parameters

    @property
    def observable_parameter_names(self) -> list:
        return self._pennylane_obs_parameters

    def draw(self, engine: str = "pennylane", **kwargs):

        if engine == "pennylane":
            # plt.figure()
            fig, ax = qml.draw_mpl(self._pennylane_circuit, **kwargs)()
            # return fig
        elif engine == "qiskit":
            return self._qiskit_circuit.draw(**kwargs)
        else:
            raise NotImplementedError("Circuit engine not implemented")

    def get_pennylane_circuit(self):
        self._pennylane_circuit = self.build_pennylane_circuit()
        return self._pennylane_circuit

    def __call__(self, *args):
        return self._pennylane_circuit(*args)

    def build_circuit_instructions(self, circuit: QuantumCircuit) -> None:

        pennylane_gates = []
        pennylane_gates_param_function = []
        pennylane_gates_wires = []
        pennylane_gates_parameters = []

        symbol_tuple = tuple([p._symbol_expr for p in circuit.parameters])

        for param in circuit.parameters:
            if param.vector.name not in pennylane_gates_parameters:
                pennylane_gates_parameters.append(param.vector.name)

        printer, modules = self._device.get_sympy_interface()

        for op in circuit.data:
            param_tuple = None
            if len(op.operation.params) >= 1:
                param_tuple = ()
                for param in op.operation.params:
                    if isinstance(param, ParameterExpression):
                        if param._symbol_expr == None:
                            param = param._coeff
                        else:
                            symbol_expr = param._symbol_expr
                            f = lambdify(
                                symbol_tuple, symbol_expr, modules=modules, printer=printer
                            )

                            param_tuple += (f,)
                    else:
                        param_tuple += (param,)

            pennylane_gates_param_function.append(param_tuple)
            pennylane_gates.append(qiskit_pennyland_gate_dict[op.operation.name])
            wires = [op.qubits[i].index for i in range(op.operation.num_qubits)]
            pennylane_gates_wires.append(wires)

        return (
            pennylane_gates,
            pennylane_gates_param_function,
            pennylane_gates_wires,
            pennylane_gates_parameters,
        )

    def build_observable_instructions(self, observable: Union[List[SparsePauliOp], SparsePauliOp]):

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

        printer, modules = self._device.get_sympy_interface()

        # Get names of all parameters in all observables
        pennylane_obs_parameters = []
        for obs in observable:
            for param in obs.parameters:
                if param.vector.name not in pennylane_obs_parameters:
                    pennylane_obs_parameters.append(param.vector.name)

        # Handle observable parameter expressions and convert them to compatible python functions

        symbol_tuple = tuple(
            sum(
                [
                    [p._symbol_expr for p in sort_parameters_after_index(obs.parameters)]
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
                        # todo check
                        coeff = coeff._coeff
                    else:
                        symbol_expr = coeff._symbol_expr
                        f = lambdify(symbol_tuple, symbol_expr, modules=modules, printer=printer)
                        pennylane_obs_param_function_.append(f)
                else:
                    pennylane_obs_param_function_.append(coeff)
            pennylane_obs_param_function.append(pennylane_obs_param_function_)

        # Convert Pauli strings into PennyLane Pauli words
        pennylane_words = []
        for obs in observable:
            pennylane_words.append(
                [pauli.string_to_pauli_word(str(p[::-1])) for p in obs._pauli_list]
            )

        if islist:
            return pennylane_obs_param_function, pennylane_words, pennylane_obs_parameters
        else:
            return pennylane_obs_param_function[0], pennylane_words[0], pennylane_obs_parameters

    @property
    def circuit_arguments(self) -> list:
        return self._pennylane_gates_parameters + self._pennylane_obs_parameters

    def build_pennylane_circuit(self):

        @self._device.add_pennylane_decorator
        def pennylane_circuit(*args):

            # list -> slow?
            circ_param_list = sum(
                [list(args[i]) for i in range(len(self._pennylane_gates_parameters))], []
            )

            obs_param_list = sum(
                [
                    list(args[len(self._pennylane_gates_parameters) + i])
                    for i in range(len(self._pennylane_obs_parameters))
                ],
                [],
            )

            # Loop through all penny lane gates
            for i, op in enumerate(self._pennylane_gates):
                if self._pennylane_gates_param_function[i] != None:
                    evaluated_param = tuple(
                        [
                            func(*circ_param_list) if callable(func) else func
                            for func in self._pennylane_gates_param_function[i]
                        ]
                    )
                    op(*evaluated_param, wires=self._pennylane_gates_wires[i])
                else:
                    op(wires=self._pennylane_gates_wires[i])

            if self._qiskit_observable == None:
                return qml.probs(wires=range(self._num_qubits))
            elif isinstance(self._qiskit_observable, list):
                expval_list = []
                for i, obs in enumerate(self._pennylane_words):
                    coeff_list = []
                    for coeff in self._pennylane_obs_param_function[i]:
                        if callable(coeff):
                            evaluated_param = coeff(*obs_param_list)
                            coeff_list.append(evaluated_param)
                        else:
                            coeff_list.append(coeff)

                    expval_list.append(qml.expval(qml.Hamiltonian(coeff_list, obs)))
                return pnp.stack(tuple(expval_list))
            else:
                coeff_list = []
                for coeff in self._pennylane_obs_param_function:
                    if callable(coeff):
                        evaluated_param = coeff(*obs_param_list)
                        coeff_list.append(evaluated_param)
                    else:
                        coeff_list.append(coeff)

                return qml.expval(qml.Hamiltonian(coeff_list, self._pennylane_words))

        return pennylane_circuit
