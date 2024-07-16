import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..encoding_circuit_base import EncodingCircuitBase


class HEE_rzrxrz(EncodingCircuitBase):
    """
    Encoding circuit with layers of RZRXRZ followed by entangling layers.

    **Example for 4 qubits and 1 layer:**

    .. plot::

        from squlearn.encoding_circuit import HEE_rzrxrz
        pqc = HEE_rzrxrz(4, 2)
        pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

    The circuit is repeated for the number of layers.

    Args:
        num_qubits (int): Number of qubits of the encoding circuit
        num_layers (int): Number of layers (default: 1)
        arrangement (str): Arrangement of the entangling gates. Options are "HEA" (default) and "ABA"
        block_width (int): Only necessary for arrangement `ABA`. Width (vertical) of each blocks for the ABA arrangement (default: 2), also refered as Nb in the paper. Must be a divisor of the number of qubits
        block_depth (int): Only necessary for arrangement `ABA`. Depth (horizontal) of each blocks for the ABA arrangement (default: 1), also refered as b in the paper.

    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 1,
        arrangement: str = "HEA",
        block_width: int = 2,  # Only necessary for ABA arrangement
        block_depth: int = 1,  # Only necessary for ABA arrangement
    ) -> None:
        super().__init__(num_qubits, 0)

        if self.num_qubits < 2:
            raise ValueError("HEE_rzrxrz requires at least two qubits.")

        self.num_layers = num_layers
        self.arrangement = arrangement
        self.block_width = block_width  # Also refered as Nb in the paper
        self.block_depth = block_depth

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the HEE_rzrxrz encoding circuit."""
        if self.arrangement == "HEA":
            num_param = 3 * self.num_qubits * self.num_layers
        elif self.arrangement == "ABA":
            num_param = 3 * self.num_qubits * self.block_depth * self.num_layers

        return num_param

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the HEE_rzrxrz encoding circuit."""
        return np.array([[-2.0 * np.pi, 2.0 * np.pi]] * self.num_parameters)

    def get_params(self) -> dict:
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
        Generates and returns the circuit of the Hubregtsen encoding circuit

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained.
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained.

        Return:
            Returns the HEE_rzrxrz encoding circuit.
        """

        def gate_block(
            QC, num_layers, num_qubits, shift_parameter=0, index_offset=0, arrangement="HEA"
        ):
            qubit_starting_index = shift_parameter
            num_qubits += shift_parameter
            for layer in range(num_layers):
                for i in range(qubit_starting_index, num_qubits):
                    if i >= self.num_qubits:
                        i -= self.num_qubits
                    QC.rz(parameters[index_offset], i)
                    QC.rx(parameters[index_offset + 1], i)
                    QC.rz(parameters[index_offset + 2], i)
                    index_offset += 3
                if arrangement == "HEA":
                    for start in (0, 1):
                        for i in range(start, num_qubits - 1, 2):
                            QC.cx(i, i + 1)

                elif arrangement == "ABA":
                    for start in (qubit_starting_index, qubit_starting_index + 1):
                        for i in range(start, num_qubits - 1, 2):
                            if i + 1 < self.num_qubits:
                                QC.cx(i, i + 1)
                            else:
                                QC.cx(i, 0)
            return QC, index_offset

        if self.num_qubits < 2:
            raise ValueError("HEE_rzrxrz requires at least two qubits.")

        QC = QuantumCircuit(self.num_qubits)
        index_offset = 0
        if self.arrangement == "HEA":
            QC = gate_block(QC, self.num_layers, self.num_qubits)[0]
        elif self.arrangement == "ABA":
            if self.num_qubits % self.block_width != 0:
                raise ValueError("Block width must be a divisor of the number of qubits.")

            number_of_blocks = int(np.ceil(self.num_qubits / self.block_width))  # vertical blocks
            shifting_factor = np.floor(self.block_width / 2)
            for layer in range(self.num_layers):
                if layer % 2 == 0:  # even layer
                    for block in range(number_of_blocks):
                        QC, index_offset = gate_block(
                            QC,
                            self.block_depth,
                            self.block_width,
                            int(block * self.block_width),
                            index_offset,
                            arrangement="ABA",
                        )
                else:
                    for block in range(number_of_blocks):
                        if (
                            shifting_factor + block * self.block_width + self.block_width
                            <= self.num_qubits
                        ):
                            QC, index_offset = gate_block(
                                QC,
                                self.block_depth,
                                self.block_width,
                                int(shifting_factor + block * self.block_width),
                                index_offset,
                                arrangement="ABA",
                            )
                        else:  # if the block is out of the range of the qubits, we need to wrap around
                            QC, index_offset = gate_block(
                                QC,
                                self.block_depth,
                                self.block_width,
                                int(shifting_factor + block * self.block_width),
                                index_offset,
                                arrangement="ABA",
                            )
                QC.barrier()

        return QC
