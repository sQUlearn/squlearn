import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit

from squlearn.util.data_preprocessing import extract_num_features

from ..encoding_circuit_base import EncodingCircuitBase


class MultiControlEncodingCircuit(EncodingCircuitBase):
    """
    Encoding circuit with HZ encoding followed by controlled Rx, Ry Rz rotations.

    **Example for 4 qubits, a 2 dimensional feature vector and 1 layer:**

    .. plot::

        from squlearn.encoding_circuit import MultiControlEncodingCircuit
        pqc = MultiControlEncodingCircuit(4, 1)
        pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10}, num_features=2)
        plt.tight_layout()

    The circuit is repeated for the number of layers.
    The circuit is closed by default, i.e. the last qubit is entangled with the first one.
    The encoding can be optionally repeated at the end to make the previous rotations not
    redundant in a fidelity kernel setting.

    Args:
        num_qubits (int): Number of qubits of the MultiControlEncodingCircuit encoding circuit
        num_layers (int): Number of layers (default: 1)
        num_features (int): Dimension of the feature vector (default: None)
        closed (bool): If true, the last and the first qubit are entangled;
                       not necessarily hardware efficient! (default: true)
        final_encoding (bool): If True, the encoding is repeated at the end (default: False)
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 1,
        num_features: int = None,
        closed: bool = True,
        final_encoding=False,
    ) -> None:
        super().__init__(num_qubits, num_features)

        if self.num_qubits < 2:
            raise ValueError("MultiControlEncodingCircuit requires at least two qubits.")

        self._num_layers = num_layers
        self._closed = closed
        self._final_encoding = final_encoding

    @property
    def num_layers(self) -> int:
        """The number of layers of the encoding circuit."""
        return self._num_layers

    @property
    def closed(self) -> bool:
        """Whether the last and the first qubit are entangled."""
        return self._closed

    @property
    def final_encoding(self) -> bool:
        """Whether the encoding is repeated at the end."""
        return self._final_encoding

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the MultiControlEncodingCircuit encoding circuit."""
        num_param = 3 * (self.num_qubits - 1) * self._num_layers
        if self._closed:
            num_param += 3 * self._num_layers
        return num_param

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the MultiControlEncodingCircuit encoding circuit."""
        return np.array([[-2.0 * np.pi, 2.0 * np.pi]] * self.num_parameters)

    @property
    def num_encoding_slots(self) -> int:
        """The number of encoding slots of the MultiControlEncodingCircuit."""
        return self.num_qubits * self._num_layers

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the MultiControlEncodingCircuit encoding circuit

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["num_layers"] = self._num_layers
        params["closed"] = self._closed
        params["final_encoding"] = self._final_encoding
        return params

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the MultiControlEncodingCircuit encoding circuit

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained.
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained.

        Return:
            Returns the circuit in Qiskit's QuantumCircuit format
        """

        if self.num_qubits < 2:
            raise ValueError("MultiControlEncodingCircuit requires at least two qubits.")

        num_features = extract_num_features(features)
        num_params = len(parameters)
        self._check_feature_encoding_slots(num_features, self.num_encoding_slots)
        self._check_feature_consistency(features)

        QC = QuantumCircuit(self.num_qubits)
        index_offset = 0
        feature_offset = 0

        for layer in range(self._num_layers):
            # First ZZ-encoding circuit
            QC.h(range(self.num_qubits))
            for i in range(self.num_qubits):
                QC.rz(features[feature_offset % num_features], i)
                feature_offset += 1

            if self._closed:
                istop = self.num_qubits
            else:
                istop = self.num_qubits - 1

            for i in range(0, istop, 2):
                QC.crx(parameters[index_offset % num_params], i, (i + 1) % self.num_qubits)
                index_offset += 1
                QC.cry(parameters[index_offset % num_params], i, (i + 1) % self.num_qubits)
                index_offset += 1
                QC.crz(parameters[index_offset % num_params], i, (i + 1) % self.num_qubits)
                index_offset += 1

            if self.num_qubits >= 2:
                if self._closed:
                    istop = self.num_qubits
                else:
                    istop = self.num_qubits - 1

                for i in range(1, istop, 2):
                    QC.crx(parameters[index_offset % num_params], i, (i + 1) % self.num_qubits)
                    index_offset += 1
                    QC.cry(parameters[index_offset % num_params], i, (i + 1) % self.num_qubits)
                    index_offset += 1
                    QC.crz(parameters[index_offset % num_params], i, (i + 1) % self.num_qubits)
                    index_offset += 1

        if self._final_encoding:
            for i in range(self.num_qubits):
                QC.rz(features[feature_offset % num_features], i)
                feature_offset += 1
        return QC
