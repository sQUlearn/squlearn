import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn import Executor
from squlearn.encoding_circuit import RandomLayeredEncodingCircuit
from qiskit.circuit import ParameterVector

from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR


class TestRandomLayeredEncodingCircuit:
    def test_random_layered_encoding_circuit_configuration(self):
        """
        The test checks if the circuit is generated correctly and uniquly for a given seed
        and set of parameters.
        """

        reference1 = [
            "Instruction(name='rz', num_qubits=1, num_clbits=0, params=[0.7853981633974483])",
            "Instruction(name='rz', num_qubits=1, num_clbits=0, params=[0.7853981633974483])",
            "Instruction(name='ry', num_qubits=1, num_clbits=0, params=[ParameterExpression(3.14159265358979*x[0])])",
            "Instruction(name='ry', num_qubits=1, num_clbits=0, params=[ParameterExpression(3.14159265358979*x[1])])",
            "Instruction(name='rz', num_qubits=1, num_clbits=0, params=[ParameterExpression(atan(x[2]))])",
            "Instruction(name='rz', num_qubits=1, num_clbits=0, params=[ParameterExpression(atan(x[3]))])",
        ]
        reference2 = [
            "Instruction(name='rz', num_qubits=1, num_clbits=0, params=[0.7853981633974483])",
            "Instruction(name='rz', num_qubits=1, num_clbits=0, params=[0.7853981633974483])",
            "Instruction(name='rz', num_qubits=1, num_clbits=0, params=[0.7853981633974483])",
            "Instruction(name='rx', num_qubits=1, num_clbits=0, params=[1.5707963267948966])",
            "Instruction(name='rx', num_qubits=1, num_clbits=0, params=[1.5707963267948966])",
            "Instruction(name='rx', num_qubits=1, num_clbits=0, params=[1.5707963267948966])",
            "Instruction(name='rz', num_qubits=1, num_clbits=0, params=[ParameterExpression(atan(x[0]))])",
            "Instruction(name='rz', num_qubits=1, num_clbits=0, params=[ParameterExpression(atan(x[1]))])",
            "Instruction(name='rz', num_qubits=1, num_clbits=0, params=[ParameterExpression(atan(x[2]))])",
        ]
        reference3 = [
            "Instruction(name='z', num_qubits=1, num_clbits=0, params=[])",
            "Instruction(name='z', num_qubits=1, num_clbits=0, params=[])",
            "Instruction(name='z', num_qubits=1, num_clbits=0, params=[])",
            "Instruction(name='cx', num_qubits=2, num_clbits=0, params=[])",
            "Instruction(name='cx', num_qubits=2, num_clbits=0, params=[])",
            "Instruction(name='rx', num_qubits=1, num_clbits=0, params=[ParameterVectorElement(x[0])])",
            "Instruction(name='rx', num_qubits=1, num_clbits=0, params=[ParameterVectorElement(x[1])])",
            "Instruction(name='rx', num_qubits=1, num_clbits=0, params=[ParameterVectorElement(x[2])])",
        ]

        pqc = RandomLayeredEncodingCircuit(num_qubits=2, max_num_layers=3)
        x = ParameterVector("x", 4)
        check_list1 = [str(op[0]) for op in pqc.get_circuit(x, [])]
        assert check_list1 == reference1

        pqc.set_params(num_qubits=3, max_num_layers=3)
        x = ParameterVector("x", 3)
        check_list2 = [str(op[0]) for op in pqc.get_circuit(x, [])]
        assert check_list2 == reference2

        pqc.set_params(seed=1234)
        check_list3 = [str(op[0]) for op in pqc.get_circuit(x, [])]
        assert check_list3 == reference3

    def test_init(self):
        circuit = RandomLayeredEncodingCircuit(
            num_qubits=2,
            min_num_layers=3,
            max_num_layers=12,
            feature_probability=0.5,
            seed=42,
        )

        assert circuit.num_qubits == 2
        assert circuit.min_num_layers == 3
        assert circuit.max_num_layers == 12
        assert circuit.feature_probability == 0.5
        assert circuit.seed == 42

    def test_get_params(self):
        circuit = RandomLayeredEncodingCircuit(
            num_qubits=2,
            min_num_layers=3,
            max_num_layers=12,
            feature_probability=0.5,
            seed=42,
        )
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": None,
            "num_qubits": 2,
            "min_num_layers": 3,
            "max_num_layers": 12,
            "feature_probability": 0.5,
            "seed": 42,
        }

    def test_set_params(self):
        circuit = RandomLayeredEncodingCircuit(
            num_qubits=2,
            min_num_layers=3,
            max_num_layers=12,
            feature_probability=0.5,
            seed=42,
        )
        circuit.set_params(
            num_qubits=3,
            min_num_layers=2,
            max_num_layers=11,
            feature_probability=0.4,
        )

        assert circuit.num_qubits == 3
        assert circuit.min_num_layers == 2
        assert circuit.max_num_layers == 11
        assert circuit.feature_probability == 0.4

    def test_get_circuit(self):
        circuit = RandomLayeredEncodingCircuit(
            num_qubits=2,
            min_num_layers=3,
            max_num_layers=12,
            feature_probability=0.5,
            seed=42,
        )
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

    def test_minimal_fit(self):
        circuit = RandomLayeredEncodingCircuit(
            num_qubits=2,
            min_num_layers=3,
            max_num_layers=12,
            feature_probability=0.5,
            seed=40,
        )

        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_train = np.array([5, 7, 9, 11, 13])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)

    def test_feature_consistency(self):
        circuit = RandomLayeredEncodingCircuit(num_qubits=4, num_features=3)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        with pytest.raises(ValueError):
            circuit.get_circuit(features, params)
