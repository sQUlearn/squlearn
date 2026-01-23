import numpy as np
from qiskit import QuantumCircuit
from squlearn.encoding_circuit import HubregtsenEncodingCircuit
from squlearn.encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from squlearn.encoding_circuit.layered_encoding_circuit import LayeredEncodingCircuit
from squlearn.encoding_circuit.pruned_encoding_circuit import (
    PrunedEncodingCircuit,
    automated_pruning,
)
from squlearn.util.executor import Executor
from tests.qiskit_circuit_equivalence import assert_circuits_equal


class TestPrunedEncodingCircuit:

    def test_num_parameters(self):
        circuit = HubregtsenEncodingCircuit(num_qubits=4, num_layers=2)
        pruned = PrunedEncodingCircuit(circuit, [0, 1, 2, 3])
        assert pruned.num_parameters == circuit.num_parameters - 4

    def test_parameter_bounds(self):
        circuit = HubregtsenEncodingCircuit(num_qubits=4, num_layers=2)
        pruned = PrunedEncodingCircuit(circuit, [0, 1, 2, 3])
        bounds = pruned.parameter_bounds
        assert bounds.shape == (pruned.num_parameters, 2)
        assert np.all(bounds == circuit.parameter_bounds[4:])

    def test_get_circuit(self):
        circuit = LayeredEncodingCircuit.from_string(
            "Ry(x)-3[Rx(p,x;=y*np.arccos(x),{y,x})-crz(p)]-Ry(p)", num_qubits=4
        )
        pruned = PrunedEncodingCircuit(circuit, [0, 1, 2, 3])
        features = np.array([0.5, -0.5])
        params = pruned.generate_initial_parameters(seed=42, num_features=4)
        qc = pruned.get_circuit(features, params)

        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 4
        # encoding slots got pruned
        assert pruned.num_encoding_slots == circuit.num_encoding_slots - 4

    def test_automated_pruning(self):
        circuit = LayeredEncodingCircuit.from_string("Rz(x)-crz(p)-Ry(x)-crx(p)", num_qubits=4)
        circuit._build_layered_pqc(4)
        pruned = automated_pruning(circuit, Executor())
        assert isinstance(pruned, PrunedEncodingCircuit)
        assert pruned.num_parameters == circuit.num_parameters - len(pruned._pruned_parameters)

    def test_pruning_removes_gates(self):

        class FeatureOnlyCircuit(EncodingCircuitBase):
            """Circuit which only contains feature-driven gates (no parameters)."""

            def __init__(self, num_qubits: int, num_features: int):
                super().__init__(num_qubits, num_features)
                self._num_features = num_features

            @property
            def num_parameters(self) -> int:
                return 0

            @property
            def num_encoding_slots(self) -> int:
                return self._num_features

            def get_circuit(self, features, parameters):
                num_features = len(features) if hasattr(features, "__len__") else 1
                qc = QuantumCircuit(self.num_qubits)
                for i in range(min(self.num_qubits, num_features)):
                    qc.rz(features[i], i)
                return qc

        class ParamOnlyCircuit(EncodingCircuitBase):
            """Circuit which only contains parameter-driven gates (no features)."""

            def __init__(self, num_qubits: int, num_params: int):
                super().__init__(num_qubits, None)
                self._num_params = num_params

            @property
            def num_parameters(self) -> int:
                return self._num_params

            @property
            def num_encoding_slots(self) -> int:
                return 0

            def get_circuit(self, features, parameters):
                num_params = len(parameters)
                qc = QuantumCircuit(self.num_qubits)
                for i in range(min(self.num_qubits, num_params)):
                    qc.ry(parameters[i], i)
                return qc

        n_qubits = 3
        n_features = 3
        n_params = 3

        feat = FeatureOnlyCircuit(num_qubits=n_qubits, num_features=n_features)
        par = ParamOnlyCircuit(num_qubits=n_qubits, num_params=n_params)

        composed = feat.compose(
            par,
            concatenate_features=False,
            concatenate_parameters=True,
            num_circuit_features=(n_features, 0),
        )

        features = np.linspace(0.1, 0.2, n_features)

        assert composed.num_parameters == par.num_parameters

        pruned_indices = list(range(0, par.num_parameters))

        pruned = PrunedEncodingCircuit(composed, pruned_indices)
        pruned_qc = pruned.get_circuit(features, np.zeros(pruned.num_parameters))
        expected_qc = feat.get_circuit(features, [])

        assert_circuits_equal(pruned_qc, expected_qc)
