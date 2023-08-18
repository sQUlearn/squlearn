import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import BlueprintCircuit
from typing import Union, Callable

from ..feature_map_base import FeatureMapBase


class QiskitFeatureMap(FeatureMapBase):
    """
    Wrapper to create sQUlearn feature maps from the `Qiskit circuit library
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
        plt = pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

    An alternative call can be made by passing the circuit library function and its arguments:

    .. code-block:: python

        from squlearn.feature_map import QiskitFeatureMap
        from qiskit.circuit.library import ZZFeatureMap
        QiskitFeatureMap(ZZFeatureMap,feature_dimension=4)

    Args:
        qiskit_circuit (Union[BlueprintCircuit, Callable,QuantumCircuit]): A Qiskit circuit or a
                                                                           Qiskit circuit library
                                                                           function
        mode (str): Option for considering the circuit parameters as features or trainable
                    parameters. Can be ``features`` or ``parameters`` or ``auto``.
                    With auto, the mode is automatically determined depending on the
                    parameter name. (default: ``auto``)
        decompose (bool): If True, the decompose method is called on the inputted circuit object.
                          (default: ``True``)
        feature_label (str): The label of the parameters that are considered as features.
                             (default: ``x``)
        parameter_label (str): The label of the parameters that are considered as trainable
                               parameters. (default: ``θp``)
        **kwargs: Arguments for the Qiskit circuit library function if it is passed as a callable
    """

    def __init__(
        self,
        qiskit_circuit: Union[BlueprintCircuit, Callable, QuantumCircuit],
        mode: str = "auto",
        decompose: bool = True,
        feature_label: str = "x",
        parameter_label: str = "θp",
        **kwargs,
    ) -> None:
        if callable(qiskit_circuit):
            self._qiskit_circuit = qiskit_circuit(**kwargs).decompose()
        else:
            if decompose:
                self._qiskit_circuit = qiskit_circuit.decompose()
            else:
                self._qiskit_circuit = qiskit_circuit

        self._num_qubits = self._qiskit_circuit.num_qubits
        self._mode = mode
        self._feature_label = feature_label
        self._parameter_label = parameter_label

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
                set_of_param_names = [p.name for p in self._qiskit_circuit.parameters]

                param_available = False
                for label in self._parameter_label:
                    if label + "[0]" in set_of_param_names:
                        param_available = True
                        break

                x_available = False
                for label in self._feature_label:
                    if label + "[0]" in set_of_param_names:
                        x_available = True
                        break

                if param_available and x_available:
                    self._num_features = 0
                    self._num_parameters = 0
                    for param in self._qiskit_circuit.parameters:
                        if True in [label in param.name for label in self._parameter_label]:
                            self._num_parameters += 1
                        elif True in [label in param.name for label in self._feature_label]:
                            self._num_features += 1
                        else:
                            raise RuntimeError("Could not assign parameter " + param.name)
                    self._mode = "both"
                elif param_available and not x_available:
                    self._num_parameters = len(self._qiskit_circuit.parameters)
                    self._num_features = 0
                    self._mode = "p"
                elif x_available and not param_available:
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

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the Qiskit feature map.

        Here arbitrarily chosen to be [-pi,pi] for all parameters.
        """
        return np.array([[-np.pi, np.pi]] * self.num_parameters)

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the Qiskit feature map

        No hyper-parameters are available for the Qiskit feature map!

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = {}
        return params

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

        if self._mode.lower() == "both":
            dictionary = {}
            i = 0
            j = 0
            for param in self._qiskit_circuit.parameters:
                if True in [label in param.name for label in self._parameter_label]:
                    dictionary[param] = parameters[i]
                    i += 1
                elif True in [label in param.name for label in self._feature_label]:
                    dictionary[param] = features[j]
                    j += 1
                else:
                    raise RuntimeError("Could not assign parameter " + param.name)
        elif self._mode.lower() == "x":
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
