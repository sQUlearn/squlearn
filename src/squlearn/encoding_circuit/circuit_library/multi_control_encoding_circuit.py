import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit

from ..encoding_circuit_base import EncodingCircuitBase


class MultiControlEncodingCircuit(EncodingCircuitBase):
    """
    Encoding circuit with HZ encoding followed by controlled Rx, Ry Rz rotations.

    **Example for 4 qubits, a 2 dimensional feature vector and 1 layer:**

    .. plot::

        from squlearn.encoding_circuit import MultiControlEncodingCircuit
        pqc = MultiControlEncodingCircuit(4, 2, 1)
        pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

    The circuit is repeated for the number of layers.
    The circuit is closed by default, i.e. the last qubit is entangled with the first one.
    The encoding can be optionally repeated at the end to make the previous rotations not
    redundant in a fidelity kernel setting.

    Args:
        num_qubits (int): Number of qubits of the MultiControlEncodingCircuit encoding circuit
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default: 1)
        closed (bool): If true, the last and the first qubit are entangled;
                       not necessarily hardware efficient! (default: true)
        final_encoding (bool): If True, the encoding is repeated at the end (default: False)
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        num_layers: int = 1,
        closed: bool = True,
        final_encoding=False,
    ) -> None:
        super().__init__(num_qubits, num_features)

        if self.num_qubits < 2:
            raise ValueError("MultiControlEncodingCircuit requires at least two qubits.")

        self.num_layers = num_layers
        self.closed = closed
        self.final_encoding = final_encoding

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the MultiControlEncodingCircuit encoding circuit."""
        num_param = 3 * (self.num_qubits - 1) * self.num_layers
        if self.closed:
            num_param += 3 * self.num_layers
        return num_param

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the MultiControlEncodingCircuit encoding circuit."""
        return np.array([[-2.0 * np.pi, 2.0 * np.pi]] * self.num_parameters)

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
        params["num_layers"] = self.num_layers
        params["closed"] = self.closed
        params["final_encoding"] = self.final_encoding
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

        nfeature = len(features)
        nparam = len(parameters)
        QC = QuantumCircuit(self.num_qubits)
        index_offset = 0
        feature_offset = 0

        for layer in range(self.num_layers):
            # First ZZ-encoding circuit
            QC.h(range(self.num_qubits))
            for i in range(self.num_qubits):
                QC.rz(features[feature_offset % nfeature], i)
                feature_offset += 1

            if self.closed:
                istop = self.num_qubits
            else:
                istop = self.num_qubits - 1

            for i in range(0, istop, 2):
                QC.crx(parameters[index_offset % nparam], i, (i + 1) % self.num_qubits)
                index_offset += 1
                QC.cry(parameters[index_offset % nparam], i, (i + 1) % self.num_qubits)
                index_offset += 1
                QC.crz(parameters[index_offset % nparam], i, (i + 1) % self.num_qubits)
                index_offset += 1

            if self.num_qubits >= 2:
                if self.closed:
                    istop = self.num_qubits
                else:
                    istop = self.num_qubits - 1

                for i in range(1, istop, 2):
                    QC.crx(parameters[index_offset % nparam], i, (i + 1) % self.num_qubits)
                    index_offset += 1
                    QC.cry(parameters[index_offset % nparam], i, (i + 1) % self.num_qubits)
                    index_offset += 1
                    QC.crz(parameters[index_offset % nparam], i, (i + 1) % self.num_qubits)
                    index_offset += 1

        if self.final_encoding:
            for i in range(self.num_qubits):
                QC.rz(features[feature_offset % nfeature], i)
                feature_offset += 1
        return QC
