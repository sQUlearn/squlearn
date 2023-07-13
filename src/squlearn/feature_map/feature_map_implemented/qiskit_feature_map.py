import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import BlueprintCircuit
from typing import Union, Callable

from ..feature_map_base import FeatureMapBase


class QiskitFeatureMap(FeatureMapBase):
    """
    Wrapper to create sQulearn feature maps from the `Qiskit circuit library
    <https://qiskit.org/documentation/apidoc/circuit_library.html>`_.

    **Example: create a feature map from Qiskit TwoLocal map**

    .. code-block:: python

       from qiskit.circuit.library import TwoLocal
       local = TwoLocal(3, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
       QiskitFeatureMap(local).draw()

    .. plot::

       from squlearn.feature_map import QiskitFeatureMap
       from qiskit.circuit.library import TwoLocal
       local = TwoLocal(3, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
       pqc = QiskitFeatureMap(local)
       plt = pqc.draw(style={'fontsize':15,'subfontsize': 10})
       plt.tight_layout()
       plt

    An alternative call can be made by passing the circuit library function and its arguments:

    .. code-block:: python

       from squlearn.feature_map import QiskitFeatureMap
       from qiskit.circuit.library import ZZFeatureMap
       QiskitFeatureMap(ZZFeatureMap,feature_dimension=4)

    Args:
        qiskit_circuit (Union[BlueprintCircuit, Callable]): A Qiskit circuit or a Qiskit
                                                            circuit library function
        mode (str): Option for considering the circuit parameters as features or trainable
                    parameters. Can be ``features`` or ``parameters`` or ``auto``.
                    With auto, the mode is automatically determined depending on the
                    parameter name. (default: ``auto``)
        **kwargs: Arguments for the Qiskit circuit library function if it is passed as a callable
    """

    def __init__(
        self,
        qiskit_circuit: Union[BlueprintCircuit, Callable],
        mode: str = "auto",
        **kwargs,
    ) -> None:
        if callable(qiskit_circuit):
            self._qiskit_circuit = qiskit_circuit(**kwargs).decompose()
        else:
            self._qiskit_circuit = qiskit_circuit.decompose()

        self._num_qubits = self._qiskit_circuit.num_qubits
        self._mode = mode

        if self._mode.lower() == "x" or self._mode.lower() == "features":
            self._num_features = len(self._qiskit_circuit.parameters)
            self._num_parameters = 0
            self._mode = "x"
        elif (
            self._mode.lower() == "p"
            or self._mode.lower() == "parameters"
            or self._mode.lower() == "param"
        ):
            self._num_parameters = len(self._qiskit_circuit.parameters)
            self._num_features = 0
            self._mode = "p"
        elif self._mode.lower() == "auto":
            if len(self._qiskit_circuit.parameters) == 0:
                self._num_features = 0
                self._num_parameters = 0
                self._mode = "empty"
            else:
                if (
                    "θ" in self._qiskit_circuit.parameters[0].name
                    or "p" in self._qiskit_circuit.parameters[0].name
                ):
                    self._num_parameters = len(self._qiskit_circuit.parameters)
                    self._num_features = 0
                    self._mode = "p"
                elif "x" in self._qiskit_circuit.parameters[0].name:
                    self._num_features = len(self._qiskit_circuit.parameters)
                    self._num_parameters = 0
                    self._mode = "x"
                else:
                    raise RuntimeError("Automatic mode determination failed!")
        else:
            raise ValueError("The type {} is not supported!".format(self._mode))

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the Qiskit feature map."""
        return self._num_parameters

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the Qiskit Feature Map

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained.
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained.

        Return:
            The circuit of the Qiskit Feature Map with the assigned parameters.
        """

        if self._mode.lower() == "x":
            dictionary = {p: v for p, v in zip(self._qiskit_circuit.parameters, features)}
            if len(self._qiskit_circuit.parameters) != len(features):
                raise ValueError("The number of features {} does not match!".format(len(features)))
        elif self._mode.lower() == "p":
            dictionary = {p: v for p, v in zip(self._qiskit_circuit.parameters, parameters)}
            if len(self._qiskit_circuit.parameters) != len(parameters):
                raise ValueError(
                    "The number of parameters {} does not match!".format(len(parameters))
                )
        elif self._mode.lower() == "empty":
            return self._qiskit_circuit
        else:
            raise ValueError("The type {} is not supported!".format(self._mode))

        return self._qiskit_circuit.assign_parameters(dictionary, inplace=False)
