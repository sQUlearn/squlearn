
from sympy import lambdify
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector,ParameterExpression

import pennylane as qml
import pennylane.pauli as pauli

from .pennylane_device import PennyLaneDevice
from .pennylane_gates import qiskit_pennyland_gate_dict


class PennyLaneCircuit():

    def __init__(self, device: PennyLaneDevice, circuit: QuantumCircuit, observable=None) -> None:

        self._device = device
        self._qiskit_circuit = circuit
        self._qiskit_observable = observable

        # Build circuit instructions for the pennylane circuit from the qiskit circuit
        self._pennylane_gates = []
        self._pennylane_gates_param_function = []
        self._pennylane_gates_wires = []
        self._build_circuit_instructions()

        # Build circuit instructions for the pennylane observable from the qiskit circuit
        self._pennylane_obs_param = []
        self._pennylane_words = []
        self._build_observable_instructions()

        self._pennylane_circuit = self.build_pennylane_circuit()

    @property
    def pennylane_circuit(self):
        return self._pennylane_circuit

    def get_pennylane_circuit(self):
        self._pennylane_circuit = self.build_pennylane_circuit()
        return self._pennylane_circuit

    def __call__(self, **kwargs):
        # TODO: Check if this works
        return self._pennylane_circuit(**kwargs)

    def _build_circuit_instructions(self) -> None:

        self._pennylane_gates = []
        self._pennylane_gates_param_function = []
        self._pennylane_gates_wires = []

        #symbol_tuple = tuple(
        #    [x_._symbol_expr for x_ in self._x_param] + [p_._symbol_expr for p_ in self._p_param]
        #)

        symbol_tuple = tuple(self._qiskit_circuit.parameters) # TODO: check if this works

        printer, modules = self._device.get_sympy_interface()

        for op in self._qiskit_circuit.data:
            param_tuple = None
            if len(op.operation.params) >= 1:
                param_tuple = ()
                for param in op.operation.params:
                    if isinstance(param, ParameterExpression):
                        if param._symbol_expr == None:
                            # todo check
                            param = param._coeff
                        else:
                            symbol_expr = param._symbol_expr
                            f = lambdify(
                                symbol_tuple, symbol_expr, modules=modules, printer=printer
                            )

                            param_tuple += (f,)
                    else:
                        param_tuple += (param,)

            self._pennylane_gates_param_function.append(param_tuple)
            self._pennylane_gates.append(qiskit_pennyland_gate_dict[op.operation.name])
            wires = [op.qubits[i].index for i in range(op.operation.num_qubits)]
            self._pennylane_gates_wires.append(wires)

    def _build_observable_instructions(self) -> None:

        if self._qiskit_observable == None:
            return None

        printer, modules = self._device.get_sympy_interface()

        self._pennylane_obs_param = []
        self._pennylane_words = [
            pauli.string_to_pauli_word(str(p[::-1])) for p in self._qiskit_observable._pauli_list
        ]

        #symbol_tuple = tuple([p_._symbol_expr for p_ in self._p_param_obs]) # TODO: fix 

        for coeff in self._qiskit_observable.coeffs:
            if isinstance(coeff, ParameterExpression):
                if coeff._symbol_expr == None:
                    # todo check
                    coeff = coeff._coeff
                else:
                    symbol_expr = coeff._symbol_expr
                    f = lambdify(symbol_tuple, symbol_expr, modules=modules, printer=printer)
                    self._pennylane_obs_param.append(f)
            else:
                self._pennylane_obs_param.append(coeff)


    def build_pennylane_circuit(self):

        # todo change parameter names
        def pennylane_circuit(x, param, param_op=None):

            circ_param_list = list(x) + list(param)
            obs_param_list = list(param_op)

            # Loop through all penny lane gates
            for i, op in enumerate(self._pennylane_gates):
                if self._pennylane_gates_param_function[i] != None:
                    evaluated_param = tuple([func(*circ_param_list) for func in self._pennylane_gates_param_function[i]])
                    op(*evaluated_param, wires=self._pennylane_gates_wires[i])
                else:
                    op(wires=self._pennylane_gates_wires[i])

            coeff_list = []
            for coeff in self._pennylane_obs_param:
                if callable(coeff):
                    evaluated_param = coeff(*obs_param_list)
                    coeff_list.append(evaluated_param)
                else:
                    coeff_list.append(coeff)

            return qml.expval(qml.Hamiltonian(coeff_list, self._pennylane_words))

        return pennylane_circuit
