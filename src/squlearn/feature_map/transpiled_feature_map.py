import numpy as np
from typing import Union, Callable

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.backend import Backend

from .feature_map_base import FeatureMapBase


class TranspiledFeatureMap(FeatureMapBase):
    """
    Class for generated a Feature Map with a transpiled circuit.

    **Example:**

    .. code-block:: python

        from squlearn.feature_map import TranspiledFeatureMap,ChebRx
        from qiskit.providers.fake_provider import FakeManilaV2

        fm = TranspiledFeatureMap(ChebRx(3,1),backend=FakeManilaV2(),initial_layout=[0,1,4])
        fm.draw()

    .. plot::

        from squlearn.feature_map import TranspiledFeatureMap,ChebRx
        from qiskit.providers.fake_provider import FakeManilaV2
        fm = TranspiledFeatureMap(ChebRx(3,1),backend=FakeManilaV2(),initial_layout=[0,1,4])
        plt = fm.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()


    Args:
        feature_map (FeatureMapBase): Feature map to be transpiled.
        backend (Backend): Backend used for the transpilation.
        transpile_func (Union[Callable,None]): Optional function for transpiling the circuit.
                                               First argument is the circuit, second the backend.
                                               If no function is specified, Qiskit's transpile
                                               function is used.
        kwargs: Additional arguments for `Qiskit's transpile function
            <https://qiskit.org/documentation/apidoc/compiler.html#qiskit.compiler.transpile>`_.

    """

    def __init__(
        self,
        feature_map: FeatureMapBase,
        backend: Backend,
        transpile_func: Union[Callable, None] = None,
        **kwargs,
    ) -> None:
        self._feature_map = feature_map
        self._backend = backend
        self._transpile_func = transpile_func

        self._x = ParameterVector("x", self._feature_map.num_features)
        self._p = ParameterVector("p", self._feature_map.num_parameters)

        self._circuit = self._feature_map.get_circuit(self._x, self._p)

        if self._transpile_func is not None:
            self._transpiled_circuit = self._transpile_func(self._circuit, self._backend)
        else:
            if "optimization_level" not in kwargs:
                kwargs["optimization_level"] = 3
            if "seed_transpiler" not in kwargs:
                kwargs["seed_transpiler"] = 0
            self._transpiled_circuit = transpile(self._circuit, self._backend, **kwargs)

        self._qubit_map = _gen_qubit_mapping(self._transpiled_circuit)
        self._kwargs = kwargs

    @property
    def num_qubits(self) -> int:
        """Number of qubits (physical) of the feature map."""
        return self._transpiled_circuit.num_qubits

    @property
    def num_physical_qubits(self) -> int:
        """Number of physical qubits of the feature map."""
        return self._transpiled_circuit.num_qubits

    @property
    def num_virtual_qubits(self) -> int:
        """Number of virtual qubits in the feature map."""
        return self._feature_map.num_qubits

    @property
    def qubit_map(self) -> dict:
        """Dictionary which maps virtual to physical qubits."""
        return self._qubit_map

    @property
    def backend(self) -> int:
        """Backend used for the transpilation."""
        return self._backend

    @property
    def num_features(self) -> int:
        """Feature dimension of the feature map."""
        return self._feature_map.num_features

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters of the feature map."""
        return self._feature_map.num_parameters

    @property
    def parameter_bounds(self) -> np.ndarray:
        """Bounds of the trainable parameters of the feature map."""
        return self._feature_map.parameter_bounds

    @property
    def feature_bounds(self) -> np.ndarray:
        """Bounds of the features of the feature map."""
        return self._feature_map.feature_bounds

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the feature map.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        return self._feature_map.get_params()

    def set_params(self, **params) -> None:
        """
        Sets value of the feature map hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. num_qubits=2
        """
        self._feature_map.set_params(**params)
        # Recompute and re-transpile the circuit by re-initializing the class
        self.__init__(self._feature_map, self._backend, self._transpile_func, **self._kwargs)

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Return the circuit of the traspiled feature map

        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                                                        from which the gate inputs are obtained
            param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                                                         from which the gate inputs are obtained

        Return:
            Returns the transpiled circuit in Qiskit's QuantumCircuit format
        """

        exchange_dict_x = dict(zip(self._x, features))
        exchange_dict_p = dict(zip(self._p, parameters))
        exchange_both = exchange_dict_x
        exchange_both.update(exchange_dict_p)
        return self._transpiled_circuit.assign_parameters(exchange_both)


def _gen_qubit_mapping(circuit: QuantumCircuit) -> dict:
    """
    Returns dictionary that maps virtual qubits to the physical ones

    Args:
        circuit (QuantumCircuit): quantum circuit (ideally transpiled)

    Returns:
        Dictionary which maps virtual to physical qubits
    """
    dic = {}
    try:
        from qiskit.transpiler.layout import TranspileLayout

        if isinstance(circuit._layout, TranspileLayout):
            layout = circuit._layout.initial_layout
        else:
            layout = circuit._layout
        bit_locations = {
            bit: {"register": register, "index": index}
            for register in layout.get_registers()
            for index, bit in enumerate(register)
        }
        for index, qubit in enumerate(layout.get_virtual_bits()):
            if qubit not in bit_locations:
                bit_locations[qubit] = {"register": None, "index": index}
        for key, val in layout.get_virtual_bits().items():
            bit_register = bit_locations[key]["register"]
            if bit_register is None or bit_register.name != "ancilla":
                dic[bit_locations[key]["index"]] = val
    except:
        for i in range(circuit.num_qubits):
            dic[i] = i
    return dic
