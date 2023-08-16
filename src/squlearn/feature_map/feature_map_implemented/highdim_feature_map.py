import numpy as np
from typing import Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info.operators import Operator

from ..feature_map_base import FeatureMapBase


class HighDimFeatureMap(FeatureMapBase):
    r"""
    The high-dimensional feature map from Reference https://doi.org/10.1103/PhysRevA.106.042431

    A feature map that can be used for the classification of high-dimensional data.

    **Example for 5 qubits, a 23 dimensional feature vector and 2 layers:**

    .. plot::

        from squlearn.feature_map import HighDimFeatureMap
        pqc = HighDimFeatureMap(5, 23, num_layers=2)
        pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

    The indexing of the feature vector can be changed by the arguments
    ``cycling``, ``cycling_type`` and ``layer_type``.

    Args:
        num_qubits (int): Number of qubits of the HighDim feature map
        num_features (int): Dimension of the feature vector
        cycling (bool): If true, the assignment of gates cycles, i.e. if reaching the last feature,
                        the layer is filled by starting again from the first feature.
                        If false, the gates are left out after reaching the last feature.
                        (default: true)
        cycling_type (str): Defines, how the indices are cycled.\n
                            ``saw``: restarts by 0, e.g. 0,1,2,3,0,1,2,3 (recommended);
                            ``hat``: goes up and then down, e.g. 0,1,2,3,2,1,0,1,2,3
        number_of_layers (int): Sets the number of layer repetitions. If not given, the number of
                                layers is determined automatically by the number of features and
                                qubits. If the given number of layers is to low, a error is thrown.
        layer_type (str): Defines in which directions the features are assigned to the gates.
                          ``columns``: iteration in columns (as shown in the example above);
                          ``rows``: iteration in rows.
        entangling_gate (str): Entangling gates that are used in the entangling layer.
                               Either ``iswap`` or ``cx`` (default: ``iswap``)
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        cycling: bool = True,
        cycling_type: str = "saw",
        num_layers: Union[None, int] = None,
        layer_type: str = "rows",
        entangling_gate: str = "iswap",
    ) -> None:
        super().__init__(num_qubits, num_features)

        self.cycling = cycling
        self.cycling_type = cycling_type
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.entangling_gate = entangling_gate

        if self.cycling_type not in ("saw", "hat"):
            raise ValueError("Unknown layer type:", self.layer_type)

        if self.layer_type not in ("columns", "rows"):
            raise ValueError("Unknown layer type:", self.layer_type)

        if self.entangling_gate not in ("cx", "iswap"):
            raise ValueError("Unknown entangling gate:", self.entangling_gate)

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the HighDim feature map (equal to 0)."""
        return 0

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the HighDim feature map

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["cycling"] = self.cycling
        params["cycling_type"] = self.cycling_type
        params["num_layers"] = self.num_layers
        params["layer_type"] = self.layer_type
        params["entangling_gate"] = self.entangling_gate
        return params

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray] = None,
    ) -> QuantumCircuit:
        """
        Returns the circuit of the HighDim feature map

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained.
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained.

        Return:
            The circuit of the high-dimensional feature map
        """

        if self.cycling_type not in ("saw", "hat"):
            raise ValueError("Unknown layer type:", self.layer_type)

        if self.layer_type not in ("columns", "rows"):
            raise ValueError("Unknown layer type:", self.layer_type)

        if self.entangling_gate not in ("cx", "iswap"):
            raise ValueError("Unknown entangling gate:", self.entangling_gate)

        def build_layer(QC: QuantumCircuit, feature_vec: ParameterVector, ioff: int):
            """
            Private function which creates a single layer
            """
            if self.layer_type == "rows":
                rows = True
            elif self.layer_type == "columns":
                rows = False
            else:
                raise ValueError("Unknown layer type:", self.layer_type)

            # Loop through all 3*n_qubit gates in this layer
            for i in range(3 * self.num_qubits):
                # Determine qubit of the gate, depending on row or column counting
                if rows:
                    iqubit = int(i / 3)
                else:
                    iqubit = i % self.num_qubits

                # Determine the index of the feature (x_i)
                ii = ioff + i
                if self.cycling:
                    if self.cycling_type == "saw":
                        ii = ii % self.num_features
                    elif self.cycling_type == "hat":  # todo better name
                        itest = ii % max(self.num_features + self.num_features - 2, 1)
                        if itest >= self.num_features:
                            ii = self.num_features + self.num_features - 2 - itest
                        else:
                            ii = itest
                    else:
                        raise ValueError("Unknown cycling type!")

                # Terminate if all features in this layer have been addressed
                if iqubit >= self.num_qubits or ii >= self.num_features:
                    break

                # Create Rz and Ry gates
                if rows:
                    if i % 3 == 0:
                        QC.rz(feature_vec[ii], iqubit)
                    elif i % 3 == 1:
                        QC.ry(feature_vec[ii], iqubit)
                    else:  # i % 3 == 2
                        QC.rz(feature_vec[ii], iqubit)
                else:
                    if int(i / self.num_qubits) == 0:
                        QC.rz(feature_vec[ii], iqubit)
                    elif int(i / self.num_qubits) == 1:
                        QC.ry(feature_vec[ii], iqubit)
                    else:  # int(i/self.num_qubits) == 2
                        QC.rz(feature_vec[ii], iqubit)

            return QC

        def entangle_layer_iswap(QC: QuantumCircuit):
            """Createn of the entangeling layer by iSWAP neighboring qubits"""

            # Manually build the iSWAP operator, since it is not available in Qiskit
            iswap_op = Operator(
                [
                    [1, 0, 0, 0],
                    [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
                    [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                    [0, 0, 0, 1],
                ]
            )

            # Build the layer
            for i in range(0, self.num_qubits - 1, 2):
                QC.unitary(iswap_op, [i, i + 1], label="iswap")
            for i in range(1, self.num_qubits - 1, 2):
                QC.unitary(iswap_op, [i, i + 1], label="iswap")

            return QC

        def entangle_layer_cx(QC: QuantumCircuit):
            """Creation of a simple nearest neighbor entangling layer"""

            for i in range(0, self.num_qubits - 1, 2):
                QC.cx(i, i + 1)
            for i in range(1, self.num_qubits - 1, 2):
                QC.cx(i, i + 1)
            return QC

        if self.num_features != len(features):
            raise ValueError("Wrong number of features")

        if parameters is not None:
            if len(parameters) != 0:
                raise ValueError("No parameters are needed!")

        # Initialize circuit and start with the Hadamard gate
        QC = QuantumCircuit(self.num_qubits)
        qubit_list = range(self.num_qubits)
        QC.h(qubit_list)

        # Determine the number of layers
        num_layers = int(self.num_features / (self.num_qubits * 3)) + 1

        # Overwrite number of layers if given by the user
        if self.num_layers is not None:
            if self.num_layers < num_layers:
                raise RuntimeError("Not all features are represented in the feature map!")
            num_layers = self.num_layers

        # Loop through the layers
        ioff = 0
        for i in range(num_layers):
            if i != 0:
                if self.entangling_gate == "iswap":
                    QC = entangle_layer_iswap(QC)
                elif self.entangling_gate == "cx":
                    QC = entangle_layer_cx(QC)
                else:
                    raise ValueError("Unknown entangling gate:", self.entangling_gate)
            QC = build_layer(QC, features, ioff)
            ioff = ioff + self.num_qubits * 3
            if self.cycling == False and ioff >= self.num_features:
                ioff = 0

        return QC
