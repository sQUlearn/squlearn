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
        reference1 = [
            "Instruction(name='z', num_qubits=1, num_clbits=0, params=[])",
            "Instruction(name='rx', num_qubits=1, num_clbits=0, params=[ParameterExpression(3.14159265358979*x[3])])",
            "Instruction(name='cx', num_qubits=2, num_clbits=0, params=[])",
            "Instruction(name='rxx', num_qubits=2, num_clbits=0, params=[ParameterExpression(p[0]*x[1])])",
            "Instruction(name='ryy', num_qubits=2, num_clbits=0, params=[ParameterExpression(p[1]*atan(x[0]))])",
            "Instruction(name='cz', num_qubits=2, num_clbits=0, params=[])",
            "Instruction(name='cy', num_qubits=2, num_clbits=0, params=[])",
            "Instruction(name='x', num_qubits=1, num_clbits=0, params=[])",
            "Instruction(name='cry', num_qubits=2, num_clbits=0, params=[ParameterExpression(3.14159265358979*x[2])])",
            "Instruction(name='rzx', num_qubits=2, num_clbits=0, params=[ParameterExpression(p[2]*atan(x[1]))])",
            "Instruction(name='cry', num_qubits=2, num_clbits=0, params=[ParameterExpression(p[3]*atan(x[0]))])",
            "Instruction(name='rxx', num_qubits=2, num_clbits=0, params=[ParameterExpression(p[4]*x[2])])",
            "Instruction(name='ryy', num_qubits=2, num_clbits=0, params=[ParameterExpression(p[5]*atan(x[3]))])",
        ]
        reference2 = [
            "Instruction(name='z', num_qubits=1, num_clbits=0, params=[])",
            "Instruction(name='rz', num_qubits=1, num_clbits=0, params=[ParameterExpression(atan(x[2]))])",
            "Instruction(name='cx', num_qubits=2, num_clbits=0, params=[])",
            "Instruction(name='rzz', num_qubits=2, num_clbits=0, params=[ParameterExpression(p[0]*x[1])])",
            "Instruction(name='rx', num_qubits=1, num_clbits=0, params=[ParameterVectorElement(x[0])])",
        ]
        reference3 = [
            "Instruction(name='rzx', num_qubits=2, num_clbits=0, params=[ParameterExpression(p[0]*atan(x[0]))])",
            "Instruction(name='rx', num_qubits=1, num_clbits=0, params=[ParameterVectorElement(x[2])])",
            "Instruction(name='rxx', num_qubits=2, num_clbits=0, params=[ParameterVectorElement(p[1])])",
            "Instruction(name='z', num_qubits=1, num_clbits=0, params=[])",
            "Instruction(name='rz', num_qubits=1, num_clbits=0, params=[ParameterExpression(atan(x[1]))])",
        ]

        # _gen_random_config must be called explictly to ensure the property num_parameters is available
        pqc = RandomEncodingCircuit(num_qubits=2, seed=2)
        x = ParameterVector("x", 4)
        pqc._gen_random_config(seed=2, num_features=len(x))
        p = ParameterVector("p", pqc.num_parameters)
        check_list1 = [str(op[0]) for op in pqc.get_circuit(x, p)]
        assert check_list1 == reference1

        pqc.set_params(num_qubits=3, min_gates=3, max_gates=5)
        x = ParameterVector("x", 3)
        pqc._gen_random_config(seed=2, num_features=len(x))
        p = ParameterVector("p", pqc.num_parameters)
        check_list2 = [str(op[0]) for op in pqc.get_circuit(x, p)]
        assert check_list2 == reference2

        pqc.set_params(seed=1234)
        pqc._gen_random_config(seed=1234, num_features=len(x))
        p = ParameterVector("p", pqc.num_parameters)
        check_list3 = [str(op[0]) for op in pqc.get_circuit(x, p)]
        assert check_list3 == reference3

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
