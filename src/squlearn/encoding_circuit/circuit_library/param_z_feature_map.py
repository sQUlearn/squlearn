import numpy as np
from typing import Union

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector
from ..encoding_circuit_base import EncodingCircuitBase


class ParamZFeatureMap(EncodingCircuitBase):
    """
    Parameterized Z feature map  with optional CNOT gates between the default layers.

    This encoding circuit is based on Qiskit's :class:`qiskit.circuit.library.ZFeatureMap`.

    **Example for 4 qubits, a 2 dimensional feature vector and 2 layers with entangling:**

    .. plot::

        from squlearn.encoding_circuit import ParamZFeatureMap
        pqc = ParamZFeatureMap(4, 2, num_layers=2, entangling=True)
        plt = pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

    Args:
        num_qubits (int): Number of qubits
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers of the encoding circuit
        entangling (bool): If true, entangling gates are added between the layers

    """

    def __init__(
        self, num_qubits: int, num_features: int, num_layers: int = 2, entangling: bool = False
    ) -> None:
        super().__init__(num_qubits, num_features)
        self._num_layers = num_layers
        self._entangling = entangling

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the encoding circuit."""
        return max(self._num_qubits, self._num_features) * self._num_layers

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the parameterized Z feature map.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["num_layers"] = self._num_layers
        params["entangling"] = self._entangling
        return params

    @property
    def num_layers(self) -> int:
        """The number of layers of the encoding circuit."""
        return self._num_layers

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the parameterized Z feature map.

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained.
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained.

        Return:
            The circuit of the parameterized Z feature map in the form of a QuantumCircuit
        """

        num_features = len(features)
        num_param = len(parameters)

        circuit = QuantumCircuit(self._num_qubits)
        index_offset = 0
        for _ in range(self._num_layers):
            for i in range(max(self._num_qubits, self._num_features)):
                if i < self._num_qubits:
                    circuit.h(i)
                circuit.p(
                    parameters[index_offset % num_param] * features[i % num_features],
                    i % self._num_qubits,
                )
                index_offset += 1

            if self._entangling:
                if self._num_layers % 2 == 0:
                    for j in range(self._num_qubits - 1):
                        circuit.cx(j, j + 1)
                else:
                    for j in range(1, self._num_qubits - 1, 2):
                        circuit.cx(j, j + 1)

        return circuit
