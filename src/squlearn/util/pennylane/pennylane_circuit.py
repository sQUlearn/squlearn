
import numpy as np
import matplotlib.pyplot as plt
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
        self._num_qubits = self._qiskit_circuit.num_qubits

        # Build circuit instructions for the pennylane circuit from the qiskit circuit
        self._pennylane_gates = []
        self._pennylane_gates_param_function = []
        self._pennylane_gates_wires = []
        self._pennylane_gates_parameters = []
        self._build_circuit_instructions()
        print("self._pennylane_gates",self._pennylane_gates)
        print("self._pennylane_gates_param_function",self._pennylane_gates_param_function)
        print("self._pennylane_gates_wires",self._pennylane_gates_wires)
        print("self._pennylane_gates_parameters",self._pennylane_gates_parameters)

        # Build circuit instructions for the pennylane observable from the qiskit circuit
        self._pennylane_obs_param_function = []
        self._pennylane_words = []
        self._pennylane_obs_parameters = []
        self._build_observable_instructions()
        print("self._pennylane_obs_param_function",self._pennylane_obs_param_function)
        print("self._pennylane_words",self._pennylane_words)
        print("self._pennylane_obs_parameters",self._pennylane_obs_parameters)

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


    def draw(self,engine:str="pennylane",**kwargs):

        if engine == "pennylane":
            #plt.figure()
            fig, ax = qml.draw_mpl(self._pennylane_circuit,**kwargs)()
            #return fig
        elif engine == "qiskit":
            return self._qiskit_circuit.draw(**kwargs)
        else:
            raise NotImplementedError("Circuit engine not implemented")

    def get_pennylane_circuit(self):
        self._pennylane_circuit = self.build_pennylane_circuit()
        return self._pennylane_circuit

    def __call__(self, *args):
        return self._pennylane_circuit(*args)

    def _build_circuit_instructions(self) -> None:

        self._pennylane_gates = []
        self._pennylane_gates_param_function = []
        self._pennylane_gates_wires = []
        self._pennylane_gates_parameters = []

        symbol_tuple = tuple([p._symbol_expr for p in self._qiskit_circuit.parameters] )

        for param in self._qiskit_circuit.parameters:
            if param.vector.name not in self._pennylane_gates_parameters:
                self._pennylane_gates_parameters.append(param.vector.name)

        printer, modules = self._device.get_sympy_interface()

        for op in self._qiskit_circuit.data:
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

            self._pennylane_gates_param_function.append(param_tuple)
            self._pennylane_gates.append(qiskit_pennyland_gate_dict[op.operation.name])
            wires = [op.qubits[i].index for i in range(op.operation.num_qubits)]
            self._pennylane_gates_wires.append(wires)

    def _build_observable_instructions(self) -> None:


        def sort_parameters_after_index(parameter_vector):
            index_list = [p.index for p in parameter_vector]
            sorted = np.argsort(index_list)
            return [parameter_vector[i] for i in sorted]

        if self._qiskit_observable == None:
            return None

        printer, modules = self._device.get_sympy_interface()

        # Get names of all parameters in the observable
        self._pennylane_obs_parameters = []
        for param in self._qiskit_observable.parameters:
            if param.vector.name not in self._pennylane_obs_parameters:
                self._pennylane_obs_parameters.append(param.vector.name)

        # Handle observable parameter expressions and convert them to compatible python functions
        self._pennylane_obs_param_function = []
        symbol_tuple = tuple([p._symbol_expr for p in sort_parameters_after_index(self._qiskit_observable.parameters)] )
        for coeff in self._qiskit_observable.coeffs:
            if isinstance(coeff, ParameterExpression):
                if coeff._symbol_expr == None:
                    # todo check
                    coeff = coeff._coeff
                else:
                    symbol_expr = coeff._symbol_expr
                    f = lambdify(symbol_tuple, symbol_expr, modules=modules, printer=printer)
                    self._pennylane_obs_param_function.append(f)
            else:
                self._pennylane_obs_param_function.append(coeff)

        # Convert Pauli strings into PennyLane Pauli words
        self._pennylane_words = [
            pauli.string_to_pauli_word(str(p[::-1])) for p in self._qiskit_observable._pauli_list
        ]

    @property
    def circuit_arguments(self) -> list:
        return self._pennylane_gates_parameters + self._pennylane_obs_parameters

    def build_pennylane_circuit(self):

        print("self._pennylane_gates_parameters",self._pennylane_gates_parameters)

        # todo change parameter names
        @self._device.add_pennylane_decorator
        #def pennylane_circuit(**kwargs):
        def pennylane_circuit(*args):
        #def pennylane_circuit(x,param,param_obs):

            # list -> slow?
            circ_param_list = sum([list(args[i]) for i in range(len(self._pennylane_gates_parameters))],[])
            obs_param_list = sum([list(args[len(self._pennylane_gates_parameters)+i]) for i in range(len(self._pennylane_obs_parameters))],[])

            #for i,p in enumerate(param):
            #    qml.RY(p, wires=i%self._num_qubits)

            # Build input parameter vector for the circuit
            # circ_param_list = []
            # for key in self._pennylane_gates_parameters:
            #     if key not in kwargs:
            #         raise ValueError("Parameter {} not found".format(key))
            #     circ_param_list += list(kwargs[key])

            # # Build input parameter vector for the observable
            # obs_param_list = []
            # for key in self._pennylane_obs_parameters:
            #     if key not in kwargs:
            #         raise ValueError("Parameter {} not found".format(key))
            #     obs_param_list += list(kwargs[key])

           # Loop through all penny lane gates
            for i, op in enumerate(self._pennylane_gates):
                if self._pennylane_gates_param_function[i] != None:
                    evaluated_param = tuple([func(*circ_param_list) if callable(func) else func for func in self._pennylane_gates_param_function[i]])
                    op(*evaluated_param, wires=self._pennylane_gates_wires[i])
                else:
                    op(wires=self._pennylane_gates_wires[i])

            # TODO: maybe change
            if self._qiskit_observable == None:
                return qml.probs(wires=range(self._num_qubits))
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
