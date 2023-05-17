import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.compiler import transpile

from .feature_map_base import FeatureMapBase


class TranspiledFeatureMap(FeatureMapBase):
    """
    Feature map base class

    Args:
        num_qubits (int): Number of Qubits of the feature map
        num_features (int): Dimension of the feature vector
    """

    def __init__(
        self,
        feature_map: FeatureMapBase,
        backend,
        transpile_func=None,
    ) -> None:
        self._feature_map = feature_map
        self._backend = backend
        self._transpile_func = transpile_func

        self.x = ParameterVector("x", self._feature_map.num_features)
        self.p = ParameterVector("p", self._feature_map.num_parameters)

        self._circuit = self._feature_map.get_circuit(self.x, self.p)

        if self._transpile_func is not None:
            self._transpiled_circuit = self._transpile_func(self._circuit, self._backend)
        else:
            self._transpiled_circuit = transpile(
                self._circuit, self._backend, optimization_level=3, seed_transpiler=0
            )

        self._qubit_map = _gen_qubit_mapping(self._transpiled_circuit)

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits of the feature map."""
        return self._feature_map.num_qubits

    @property
    def num_all_qubits(self) -> int:
        """Returns the number of qubits of the feature map."""
        return self._transpiled_circuit.num_qubits

    @property
    def qubit_map(self):
        return self._qubit_map

    @property
    def backend(self) -> int:
        """Returns the dimension of features of the feature map."""
        return self._backend

    @property
    def num_features(self) -> int:
        """Returns the dimension of features of the feature map."""
        return self._feature_map.num_features

    @property
    def num_parameters(self) -> int:
        """Returns the number of trainable parameters of the feature map."""
        return self._feature_map.num_parameters

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Return the circuit feature map (has to be overwritten, otherwise a NotImplementedError is thrown)

        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                from which the gate inputs are obtained
            param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                from which the gate inputs are obtained

        Return:
            Returns the circuit in qiskit QuantumCircuit format
        """

        exchange_dict_x = dict(zip(self.x, features))
        exchange_dict_p = dict(zip(self.p, parameters))
        exchange_both = exchange_dict_x
        exchange_both.update(exchange_dict_p)
        return self._transpiled_circuit.assign_parameters(exchange_both)


def _gen_qubit_mapping(circuit):
    """
    Returns dictionary that maps virtual qubits to the physical ones

    Args:
        circ : quantum circuit (ideally transpiled)

    Returns:
        Dictionary which maps virtual to physical qubits
    """
    dic = {}
    try:
        from qiskit.transpiler.layout import TranspileLayout

        if isinstance(circuit._layout, TranspileLayout):
            layout = circuit._layout.initial_layout
        else:
            layout = circuit._layout
        bit_locations = {
            bit: {"register": register, "index": index}
            for register in layout.get_registers()
            for index, bit in enumerate(register)
        }
        for index, qubit in enumerate(layout.get_virtual_bits()):
            if qubit not in bit_locations:
                bit_locations[qubit] = {"register": None, "index": index}
        for key, val in layout.get_virtual_bits().items():
            bit_register = bit_locations[key]["register"]
            if bit_register is None or bit_register.name != "ancilla":
                dic[bit_locations[key]["index"]] = val
    except:
        for i in range(circuit.num_qubits):
            dic[i] = i
    return dic
