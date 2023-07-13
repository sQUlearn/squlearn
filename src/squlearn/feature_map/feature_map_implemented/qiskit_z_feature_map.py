import numpy as np
from typing import Union, Optional, Callable
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap

from ..feature_map_base import FeatureMapBase


class QiskitZFeatureMap(FeatureMapBase):
    """
    Wrapper to use Qiskit's ZFeatureMap

    **Example for 4 qubits and 2 layers:**

    .. plot::

       from squlearn.feature_map import QiskitZFeatureMap
       pqc = QiskitZFeatureMap(4, reps=2)
       plt = pqc.draw(style={'fontsize':15,'subfontsize': 10})
       plt.tight_layout()
       plt

    Number of features is equal to the number of qubits.

    Args:
        num_qubits: The number of qubits
        reps: The number of repeated circuits. Defaults to 2, has a minimum value of 1
        data_map_func: A mapping function for data x which can be supplied to override the default mapping from self_product()
        insert_barriers: If True, barriers are inserted in between the evolution instructions and hadamard layers
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 2,
        data_map_func: Optional[Callable[[np.ndarray], float]] = None,
        insert_barriers: bool = False,
    ) -> None:
        super().__init__(num_qubits, num_qubits)

        self._reps = reps
        self._data_map_func = data_map_func
        self._insert_barriers = insert_barriers

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the Qiskit ZFeatureMap (equal to 0)."""
        return 0

    @property
    def num_layers(self) -> int:
        """The number of layers of the Qiskit ZFeatureMap."""
        return self._reps

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the Qiskit ZFeatureMap

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained.
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained.

        Return:
            The circuit of the Qiskit ZFeatureMap
        """
        circuit_z_feature_map = ZFeatureMap(
            feature_dimension=self._num_features,
            reps=self._reps,
            data_map_func=self._data_map_func,
            parameter_prefix="xx",
            insert_barriers=self._insert_barriers,
        ).decompose()

        if len(circuit_z_feature_map.parameters) != len(features):
            raise ValueError("The number of features {} does not match!".format(len(features)))

        dictionary = {p: v for p, v in zip(circuit_z_feature_map.parameters, features)}
        circuit_z_feature_map.assign_parameters(dictionary, inplace=True)

        return circuit_z_feature_map
