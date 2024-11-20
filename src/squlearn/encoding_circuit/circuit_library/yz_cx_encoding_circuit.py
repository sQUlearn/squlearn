import numpy as np
from typing import Union
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector

from ..encoding_circuit_base import EncodingCircuitBase


class YZ_CX_EncodingCircuit(EncodingCircuitBase):
    """
    Creates the YZ-CX Encoding Circuit from reference [1].

    **Example for 4 qubits, a 4 dimensional feature vector, 2 layers and c = 2.0:**

    .. plot::

        from squlearn.encoding_circuit import YZ_CX_EncodingCircuit
        pqc = YZ_CX_EncodingCircuit(4, 4, 2, c=2.0)
        plt = pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

    One combination of Ry and Rz is considered as a single layer.

    Args:
        num_qubits (int): Number of qubits of the YZ-CX Encoding Circuit encoding circuit
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default: 1)
        c (float): Prefactor :math:`c` for rescaling the data (default: 1.0)

    References
    ----------
    [1]: T. Haug, C. N. Self and M. S. Kim, "Quantum machine learning of large datasets using
    randomized measurements", `arxiv:2108.01039v3 (2021). <https://arxiv.org/abs/2108.01039v3>`_
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        num_layers: int = 1,
        closed: bool = True,
        c: float = 1.0,
    ) -> None:
        super().__init__(num_qubits, num_features)
        self.closed = closed

        self._num_layers = num_layers
        self._c = c

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the YZ-CX Encoding Circuit encoding circuit."""
        return 2 * self.num_qubits * self._num_layers

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the YZ-CX Encoding Circuit encoding circuit."""
        return np.array([[-np.pi, np.pi]] * self.num_parameters)

    @property
    def num_layers(self) -> int:
        """The number of layers of the YZ-CX Encoding Circuit encoding circuit."""
        return self._num_layers

    @property
    def c(self) -> int:
        """The prefactor :math:`c` of the YZ-CX Encoding Circuit encoding circuit."""
        return self._c

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the YZ-CX Encoding Circuit encoding circuit

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["num_layers"] = self._num_layers
        params["c"] = self._c
        return params

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Return the circuit of the YZ-CX encoding circuit.

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained.
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained.

        Return:
            Returns the circuit in qiskit format.
        """

        nfeature = len(features)
        nparam = len(parameters)

        # Creates the layers of the encoding circuit
        QC = QuantumCircuit(self.num_qubits)
        index_offset = 0
        feature_offset = 0
        for layer in range(self.num_layers):
            for i in range(self.num_qubits):
                QC.ry(
                    parameters[index_offset % nparam]
                    + self.c * features[feature_offset % nfeature],
                    i,
                )
                index_offset += 1
                QC.rz(
                    parameters[index_offset % nparam]
                    + self.c * features[feature_offset % nfeature],
                    i,
                )
                index_offset += 1
                feature_offset += 1
            # Entangling layer depends on odd or even layer
            if self.num_qubits >= 2:
                for i in range(layer % 2, self.num_qubits + self.closed - 1, 2):
                    QC.cx(i, (i + 1) % self.num_qubits)
        return QC
