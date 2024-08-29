import pytest

from squlearn.encoding_circuit import RandomLayeredEncodingCircuit

from qiskit.circuit import ParameterVector


class TestRandomEncodingCircuits:

    def test_random_encoding_circuit(self):

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

        pqc = RandomLayeredEncodingCircuit(num_qubits=2, num_features=4, max_num_layers=3)
        x = ParameterVector("x", 4)
        check_list1 = [str(op[0]) for op in pqc.get_circuit(x, [])]
        assert check_list1 == reference1

        pqc.set_params(num_qubits=3, num_features=3, max_num_layers=3)
        x = ParameterVector("x", 3)
        check_list2 = [str(op[0]) for op in pqc.get_circuit(x, [])]
        assert check_list2 == reference2

        pqc.set_params(seed=1234)
        check_list3 = [str(op[0]) for op in pqc.get_circuit(x, [])]
        assert check_list3 == reference3
