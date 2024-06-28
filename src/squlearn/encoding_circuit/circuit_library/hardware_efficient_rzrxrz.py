import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..encoding_circuit_base import EncodingCircuitBase


class HEE_rzrxrz(EncodingCircuitBase):
    """
    Encoding circuit with layers of RZRXRZ followed by entangling layers.

    **Example for 4 qubits, a 2 dimensional feature vector and 1 layer:**

    .. plot::

        from squlearn.encoding_circuit import MultiControlEncodingCircuit
        pqc = HEE_rzrxrz(4, 2, 1)
        pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

    The circuit is repeated for the number of layers.

    Args:
        num_qubits (int): Number of qubits of the encoding circuit
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default: 1)
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__(num_qubits, num_features)

        if self.num_qubits < 2:
            raise ValueError("HEE_rzrxrz requires at least two qubits.")

        self.num_layers = num_layers

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the HEE_rzrxrz encoding circuit."""
        num_param = 3 * (self.num_qubits) * self.num_layers
        
        return num_param

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the HEE_rzrxrz encoding circuit."""
        return np.array([[-2.0 * np.pi, 2.0 * np.pi]] * self.num_parameters)

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the HEE_rzrxrz encoding circuit

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["num_layers"] = self.num_layers
        return params

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the HEE_rzrxrz encoding circuit

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained.
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained.

        Return:
            Returns the circuit in Qiskit's QuantumCircuit format
        """

        if self.num_qubits < 2:
            raise ValueError("HEE_rzrxrz requires at least two qubits.")

        nfeature = len(features)
        nparam = len(parameters)
        QC = QuantumCircuit(self.num_qubits)
        index_offset = 0

        for layer in range(self.num_layers):
            for i in range(self.num_qubits):
                QC.rz(parameters[index_offset], i)
                QC.rx(parameters[index_offset + 1], i)
                QC.rz(parameters[index_offset + 2], i)
                index_offset += 3
            QC.barrier()
            for i in range(0, self.num_qubits - 1, 2):
                QC.cx(i, i + 1)
            for i in range(1, self.num_qubits - 1, 2):
                QC.cx(i, i + 1)
                
        return QC