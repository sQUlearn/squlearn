import pytest
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from squlearn.feature_map import LayeredFeatureMap, ChebPQC


class TestLayeredFeatureMap:
    """ Test class for LayeredFeatureMap."""

    def test_layered_feature_map_gates(self):
        """ Test the non-parameterized gates of the LayeredFeatureMap."""

        lfm = LayeredFeatureMap(num_qubits=4, num_features=0)

        # Test the H gate
        lfm.H()
        expected_circuit = QuantumCircuit(4)
        expected_circuit.h(range(4))
        assert str(lfm.get_circuit([], [])) == str(expected_circuit)

        # Test the X gate
        lfm.X()
        expected_circuit.x(range(4))
        assert str(lfm.get_circuit([], [])) == str(expected_circuit)

        # Test the Y gate
        lfm.Y()
        expected_circuit.y(range(4))
        assert str(lfm.get_circuit([], [])) == str(expected_circuit)

        # Test the Z gate
        lfm.Z()
        expected_circuit.z(range(4))
        assert str(lfm.get_circuit([], [])) == str(expected_circuit)

        # Test the I gate
        lfm.I()
        expected_circuit.i(range(4))
        assert str(lfm.get_circuit([], [])) == str(expected_circuit)

        # Test the S gate
        lfm.S()
        expected_circuit.s(range(4))
        assert str(lfm.get_circuit([], [])) == str(expected_circuit)

        # Test the Sdg gate
        lfm.S_conjugate()
        expected_circuit.sdg(range(4))
        assert str(lfm.get_circuit([], [])) == str(expected_circuit)

        # Test the T gate
        lfm.T()
        expected_circuit.t(range(4))
        assert str(lfm.get_circuit([], [])) == str(expected_circuit)

        # Test the Tdg gate
        lfm.T_conjugate()
        expected_circuit.tdg(range(4))
        assert str(lfm.get_circuit([], [])) == str(expected_circuit)

    def test_layered_feature_map_param_gates(self):
        """ Test the parameterized gates of the LayeredFeatureMap."""

        lfm = LayeredFeatureMap(num_qubits=4, num_features=0)
        expected_circuit = QuantumCircuit(4)
        p = ParameterVector("p", 28)

        # Test the RX gate
        lfm.Rx("p", encoding=np.arccos)
        for i in range(4):
            expected_circuit.rx(np.arccos(p[i]), i)

        assert str(lfm.get_circuit([], p)) == str(expected_circuit)

        # Test the RY gate
        lfm.Ry("p", encoding=np.arccos)
        for i in range(4):
            expected_circuit.ry(np.arccos(p[i + 4]), i)
        assert str(lfm.get_circuit([], p)) == str(expected_circuit)

        # Test the RZ gate
        lfm.Rz("p", encoding=np.arccos)
        for i in range(4):
            expected_circuit.rz(np.arccos(p[i + 8]), i)
        assert str(lfm.get_circuit([], p)) == str(expected_circuit)

        # Test the Phase gate
        lfm.P("p", encoding=np.arccos)
        for i in range(4):
            expected_circuit.p(np.arccos(p[i + 12]), i)
        assert str(lfm.get_circuit([], p)) == str(expected_circuit)

        # Test the U gate
        lfm.U(("p", "p", "p"))
        ioff = 16
        for i in range(4):
            expected_circuit.u(p[ioff], p[ioff + 1], p[ioff + 2], i)
            ioff += 3

        assert str(lfm.get_circuit([], p)) == str(expected_circuit)

    def test_layered_feature_map_entangling_gates(self):
        """ Test the entangling gates of the LayeredFeatureMap. """

        def add_NN(gate_function):
            gate_function(0, 1)
            gate_function(2, 3)
            gate_function(1, 2)

        def add_AA(gate_function):
            gate_function(0, 1)
            gate_function(0, 2)
            gate_function(0, 3)
            gate_function(1, 2)
            gate_function(1, 3)
            gate_function(2, 3)

        # Create a LayeredFeatureMap with 2 layers and 3 features per layer
        lfm = LayeredFeatureMap(num_qubits=4, num_features=0)
        expected_circuit = QuantumCircuit(4)

        lfm_list = [
            lfm.ch_entangling,
            lfm.cx_entangling,
            lfm.cy_entangling,
            lfm.cz_entangling,
            lfm.swap,
        ]
        qiskit_list = [
            expected_circuit.ch,
            expected_circuit.cx,
            expected_circuit.cy,
            expected_circuit.cz,
            expected_circuit.swap,
        ]

        for lfm_gate, qiskit_gate in zip(lfm_list, qiskit_list):
            lfm_gate("NN")
            add_NN(qiskit_gate)
            assert str(lfm.get_circuit([], [])) == str(expected_circuit)

            lfm_gate("AA")
            add_AA(qiskit_gate)
            assert str(lfm.get_circuit([], [])) == str(expected_circuit)

    def test_layered_feature_map_param_entangling_gates(self):
        """ Test the parameterized entangling gates of the LayeredFeatureMap. """

        def add_NN(gate_function, p, offset=0):
            gate_function(np.arccos(p[offset]), 0, 1)
            gate_function(np.arccos(p[offset + 1]), 2, 3)
            gate_function(np.arccos(p[offset + 2]), 1, 2)
            return offset + 3

        def add_AA(gate_function, p, offset=0):
            gate_function(np.arccos(p[offset]), 0, 1)
            gate_function(np.arccos(p[offset + 1]), 0, 2)
            gate_function(np.arccos(p[offset + 2]), 0, 3)
            gate_function(np.arccos(p[offset + 3]), 1, 2)
            gate_function(np.arccos(p[offset + 4]), 1, 3)
            gate_function(np.arccos(p[offset + 5]), 2, 3)
            return offset + 6

        # Create a LayeredFeatureMap with 2 layers and 3 features per layer
        lfm = LayeredFeatureMap(num_qubits=4, num_features=0)
        expected_circuit = QuantumCircuit(4)
        p = ParameterVector("p", 72)
        offset = 0

        lfm_list = [
            lfm.cp_entangling,
            lfm.crx_entangling,
            lfm.cry_entangling,
            lfm.crz_entangling,
            lfm.rxx_entangling,
            lfm.ryy_entangling,
            lfm.rzz_entangling,
            lfm.rzx_entangling,
        ]
        qiskit_list = [
            expected_circuit.cp,
            expected_circuit.crx,
            expected_circuit.cry,
            expected_circuit.crz,
            expected_circuit.rxx,
            expected_circuit.ryy,
            expected_circuit.rzz,
            expected_circuit.rzx,
        ]

        for lfm_gate, qiskit_gate in zip(lfm_list, qiskit_list):
            lfm_gate("p", ent_strategy="NN", encoding=np.arccos)
            offset = add_NN(qiskit_gate, p, offset)
            assert str(lfm.get_circuit([], p)) == str(expected_circuit)

            lfm_gate("p", ent_strategy="AA", encoding=np.arccos)
            offset = add_AA(qiskit_gate, p, offset)
            assert str(lfm.get_circuit([], p)) == str(expected_circuit)

    def test_from_string(self):
        """ Test the from_string method of the LayeredFeatureMap. """

        lfm = LayeredFeatureMap.from_string(
            "Ry(p)-3[Rx(p,x;=y*np.arccos(x),{y,x})-crz(p)]-Ry(p)", num_qubits=4, num_features=1
        )

        cpqc = ChebPQC(num_qubits=4, num_features=1, num_layers=3, closed=False)

        assert str(lfm.draw(output="text")) == str(cpqc.draw(output="text"))
