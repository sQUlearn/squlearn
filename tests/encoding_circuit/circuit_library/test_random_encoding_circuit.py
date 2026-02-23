import numpy as np
import pytest

from qiskit import QuantumCircuit
from squlearn import Executor
from squlearn.encoding_circuit import RandomEncodingCircuit
from qiskit.circuit import ParameterVector

from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR


class TestRandomEncodingCircuit:
    def test_random_encoding_circuit_configuration(self):
        """
        The test checks if the circuit is generated correctly and uniquly for a given seed
        and set of parameters.
        """
        reference1 = (
            "     ┌───┐┌────────────┐     ┌─────────────────┐┌───────────────────────┐        ┌──"
            "─┐              ┌───────────────────────┐                       ┌─────────────────┐┌"
            "───────────────────────┐\nq_0: ┤ Z ├┤ Rx(π*x[3]) ├──■──┤0                ├┤0        "
            "              ├─■───■──┤ X ├──────■───────┤1                      ├───────────■─────"
            "──────┤0                ├┤0                      ├\n     └───┘└────────────┘┌─┴─┐│  "
            "Rxx(p[0]*x[1]) ││  Ryy(p[1]*atan(x[0])) │ │ ┌─┴─┐└───┘┌─────┴──────┐│  Rzx(p[2]*atan"
            "(x[1])) │┌──────────┴──────────┐│  Rxx(p[4]*x[2]) ││  Ryy(p[5]*atan(x[3])) │\nq_1: ─"
            "──────────────────┤ X ├┤1                ├┤1                      ├─■─┤ Y ├─────┤ Ry"
            "(π*x[2]) ├┤0                      ├┤ Ry(p[3]*atan(x[0])) ├┤1                ├┤1     "
            "                 ├\n                        └───┘└─────────────────┘└───────────────"
            "────────┘   └───┘     └────────────┘└───────────────────────┘└─────────────────────┘"
            "└─────────────────┘└───────────────────────┘"
        )
        reference2 = (
            "           ┌───┐                                        \nq_0: ──────┤ Z ├─────────■"
            "──────────────────────────────\n     ┌─────┴───┴──────┐  │                          "
            "    \nq_1: ┤ Rz(atan(x[2])) ├──┼───■──────────────────────────\n     └──────────────"
            "──┘┌─┴─┐ │ZZ(p[0]*x[1]) ┌──────────┐\nq_2: ──────────────────┤ X ├─■──────────────┤ "
            "Rx(x[0]) ├\n                       └───┘                └──────────┘"
        )
        reference3 = (
            "     ┌───────────────────────┐            ┌────────────┐┌────────────────┐\nq_0: ┤1 "
            "                     ├────────────┤0           ├┤ Rz(atan(x[1])) ├\n     │          "
            "             │   ┌───┐    │            │└────────────────┘\nq_1: ┤  Rzx(p[0]*atan(x["
            "0])) ├───┤ Z ├────┤  Rxx(p[1]) ├──────────────────\n     │                       │┌─"
            "─┴───┴───┐│            │                  \nq_2: ┤0                      ├┤ Rx(x[2])"
            " ├┤1           ├──────────────────\n     └───────────────────────┘└──────────┘└─────"
            "───────┘                  "
        )

        # _gen_random_config must be called explictly to ensure the property num_parameters is available
        pqc = RandomEncodingCircuit(num_qubits=2, seed=2)
        x = ParameterVector("x", 4)
        pqc._gen_random_config(seed=2, num_features=len(x))
        p = ParameterVector("p", pqc.num_parameters)
        check_circuit1 = repr(pqc.draw("text", fold=-1, num_features=4))
        assert check_circuit1 == reference1

        pqc.set_params(num_qubits=3, min_gates=3, max_gates=5)
        x = ParameterVector("x", 3)
        pqc._gen_random_config(seed=2, num_features=len(x))
        p = ParameterVector("p", pqc.num_parameters)
        check_circuit2 = repr(pqc.draw("text", fold=-1, num_features=3))
        assert check_circuit2 == reference2

        pqc.set_params(seed=1234)
        pqc._gen_random_config(seed=1234, num_features=len(x))
        p = ParameterVector("p", pqc.num_parameters)
        check_circuit3 = repr(pqc.draw("text", fold=-1, num_features=3))
        assert check_circuit3 == reference3

    def test_init(self):
        circuit = RandomEncodingCircuit(num_qubits=2, min_gates=9, max_gates=40, seed=42)

        assert circuit.num_qubits == 2
        assert circuit.min_gates == 9
        assert circuit.max_gates == 40
        assert circuit.seed == 42

    def test_get_params(self):
        circuit = RandomEncodingCircuit(num_qubits=2, min_gates=9, max_gates=40, seed=42)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": None,
            "num_qubits": 2,
            "min_gates": 9,
            "max_gates": 40,
            "seed": 42,
            "encoding_weights": circuit.encoding_weights,
            "gate_weights": circuit.gate_weights,
        }

    def test_set_params(self):
        circuit = RandomEncodingCircuit(num_qubits=2, min_gates=9, max_gates=40, seed=42)
        circuit.set_params(
            num_qubits=3,
            min_gates=11,
            max_gates=35,
        )
        assert circuit.num_qubits == 3
        assert circuit.min_gates == 11
        assert circuit.max_gates == 35

        with pytest.raises(ValueError):
            circuit.set_params(invalid_param="invalid")

    def test_get_circuit(self):
        circuit = RandomEncodingCircuit(num_qubits=2, min_gates=9, max_gates=40, seed=42)
        circuit._gen_random_config(seed=42, num_features=2)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

    def test_minimal_fit(self):
        circuit = RandomEncodingCircuit(num_qubits=2, min_gates=9, max_gates=40, seed=40)

        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_train = np.array([5, 7, 9, 11, 13])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)

    def test_feature_consistency(self):
        circuit = RandomEncodingCircuit(num_qubits=4, num_features=3)
        features = np.array([0.5, -0.5])

        with pytest.raises(ValueError):
            circuit.get_circuit(features, [])
