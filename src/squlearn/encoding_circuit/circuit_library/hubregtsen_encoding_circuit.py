import numpy as np
from typing import Union
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector

from ..encoding_circuit_base import EncodingCircuitBase


class HubregtsenEncodingCircuit(EncodingCircuitBase):
    """
    Creates the data reuploading encoding circuit as presented in reference [1].

    **Example for 4 qubits, a 2 dimensional feature vector, 2 layers:**

    .. plot::

        from squlearn.encoding_circuit import HubregtsenEncodingCircuit
        pqc = HubregtsenEncodingCircuit(4, 2, 2)
        plt = pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

    The encoding can be optionally repeated at the end to make the previous rotations not
    redundant in a fidelity kernel setting.
    The circuit is closed by default, i.e. the last qubit is entangled with the first one.

    Args:
        num_qubits (int): Number of qubits of the encoding circuit
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default:1)
        closed (bool): If true, the last and the first qubit are entangled;
                       not necessarily hardware efficient! (default: true)
        final_encoding (bool): If True, the encoding is repeated at the end (default: False)

    References
    ----------
    [1]: T. Hubregtsen et al., "Training Quantum Embedding Kernels on Near-Term Quantum Computers",
    `arXiv:2105.02276v1 (2021). <https://arxiv.org/abs/2105.02276>`_
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

        self.num_layers = num_layers
        self.closed = closed
        self.final_encoding = final_encoding

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the Hubregtsen encoding circuit."""
        num_param = self.num_qubits * self.num_layers
        if self.num_qubits > 2:
            if self.closed:
                num_param += self.num_qubits * self.num_layers
            else:
                num_param += (self.num_qubits - 1) * self.num_layers
        return num_param

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the Hubregtsen encoding circuit."""

        bound_array = np.zeros((self.num_parameters, 2))
        # Single theta Ry gates
        index_offset = 0
        for layer in range(self.num_layers):
            for i in range(self.num_qubits):
                bound_array[index_offset] = [-np.pi, np.pi]
                index_offset += 1

            # Entangled theta CRZ gates
            if self.num_qubits > 2:
                if self.closed:
                    istop = self.num_qubits
                else:
                    istop = self.num_qubits - 1

                for i in range(istop):
                    bound_array[index_offset] = [-2.0 * np.pi, 2.0 * np.pi]
                    index_offset += 1
        return bound_array

    @property
    def feature_bounds(self) -> np.ndarray:
        """The bounds of the features of the Hubregtsen encoding circuit."""
        return np.array([[-np.pi, np.pi]] * self.num_features)

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the Hubregtsen encoding circuit

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
        Generates and returns the circuit of the Hubregtsen encoding circuit

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained.
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained.

        Return:
            Returns the Hubregtsen circuit in qiskit QuantumCircuit format
        """

        nfeatures = len(features)
        nparam = len(parameters)

        QC = QuantumCircuit(self.num_qubits)
        index_offset = 0

        QC.h(range(self.num_qubits))

        # Loops through the layers
        for layer in range(self.num_layers):
            # Loops through the data encoding gates
            n_feature_loop = int(np.ceil(self.num_features / self.num_qubits))
            for i in range(n_feature_loop * self.num_qubits):
                if (i // self.num_qubits) % 2 == 0:
                    QC.rz(features[i % nfeatures], i % self.num_qubits)
                else:
                    QC.rx(features[i % nfeatures], i % self.num_qubits)

            # Single theta Ry gates
            for i in range(self.num_qubits):
                QC.ry(parameters[index_offset % nparam], i)
                index_offset += 1

            # Entangled theta CRZ gates
            if self.num_qubits > 2:
                if self.closed:
                    istop = self.num_qubits
                else:
                    istop = self.num_qubits - 1

                for i in range(istop):
                    QC.crz(parameters[index_offset % nparam], i, (i + 1) % self.num_qubits)
                    index_offset += 1

        if self.final_encoding:
            # Repeat encoding finally to make the previous rotations not redundant
            n_feature_loop = int(np.ceil(self.num_features / self.num_qubits))
            for i in range(n_feature_loop * self.num_qubits):
                if int(np.ceil(i / self.num_qubits)) % 2 == 0:
                    QC.rz(features[i % nfeatures], i % self.num_qubits)
                else:
                    QC.rx(features[i % nfeatures], i % self.num_qubits)

        return QC
