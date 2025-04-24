import numpy as np
from qiskit import QuantumCircuit
from squlearn.encoding_circuit import HubregtsenEncodingCircuit
from squlearn.encoding_circuit.layered_encoding_circuit import Layer, LayeredEncodingCircuit
from squlearn.encoding_circuit.pruned_encoding_circuit import (
    PrunedEncodingCircuit,
    automated_pruning,
)
from squlearn.util.executor import Executor


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
        # encdoding slots got pruned
        assert pruned.num_encoding_slots == circuit.num_encoding_slots - 4


def test_automated_pruning():
    circuit_before_pruning = LayeredEncodingCircuit(2)
    layer = Layer(circuit_before_pruning)
    layer.Rz("p")
    layer.Ry("x")
    layer.Z()
    layer.Ry("p")
    layer.Rz("p")
    circuit_before_pruning.add_layer(layer)
    circuit_before_pruning._build_layered_pqc(4)

    auto_pruned = automated_pruning(circuit_before_pruning, Executor("pennylane"), n_sample=3)
    assert isinstance(auto_pruned, PrunedEncodingCircuit)

    features = np.array([0.5, -0.5])
    params = auto_pruned.generate_initial_parameters(seed=42, num_features=4)

    qc = auto_pruned.get_circuit(features, params)
    assert isinstance(qc, QuantumCircuit)

    # two parameters were pruned
    assert len(auto_pruned._pruned_parameters) == 2
    assert auto_pruned.num_parameters == circuit_before_pruning.num_parameters - 2
