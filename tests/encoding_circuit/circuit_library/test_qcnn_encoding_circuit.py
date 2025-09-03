import pytest
import numpy as np
from qiskit import QuantumCircuit
from squlearn.encoding_circuit import QCNNEncodingCircuit
from squlearn.observables.observable_base import ObservableBase


class TestQCNNEncodingCircuit:

    def test_init(self):
        num_qubits = 2
        circuit = QCNNEncodingCircuit(num_qubits=num_qubits)
        assert circuit.num_qubits == num_qubits
        assert circuit._default is False
        assert circuit.num_parameters == 0
        assert circuit.operations_list == []
        assert circuit.left_qubits == list(range(num_qubits))
        assert circuit.num_encoding_slots == np.inf

    def test_init_raises_on_zero_qubits_with_default_true(self):
        with pytest.raises(ValueError):
            QCNNEncodingCircuit(num_qubits=0, default=True)

    def test_default_init(self):
        circuit = QCNNEncodingCircuit(num_qubits=3, default=True)

        # basic properties
        assert isinstance(circuit.operations_list, list)
        assert circuit.num_qubits == 3
        assert circuit._default is True

        # operations_list should not be empty and last operation must be FC
        assert len(circuit.operations_list) >= 1
        assert circuit.operations_list[-1][0] == "FC"

        # after default build we expect the circuit reduced to a single left qubit for 3->1 pooling
        assert isinstance(circuit.left_qubits, list)
        assert len(circuit.left_qubits) == 1
        assert circuit.left_qubits == [0]

        # num_parameters should be non-negative and consistent with operations_list
        assert isinstance(circuit.num_parameters, int)
        assert circuit.num_parameters >= 0

    def test_default_one_qubit(self):
        circuit = QCNNEncodingCircuit(num_qubits=1, default=True)
        assert len(circuit.operations_list) >= 1
        # first operation should be FC and since only one qubit exists it should be the main op
        assert circuit.operations_list[0][0] == "FC"
        # left_qubits remains a list of the single qubit index
        assert circuit.left_qubits == [0]

    def test_get_circuit_zero_qubits(self):
        circuit = QCNNEncodingCircuit(num_qubits=0, default=False)
        with pytest.raises(ValueError):
            circuit.get_circuit(np.array([]), np.array([]))

    def test_get_circuit(self):
        circuit = QCNNEncodingCircuit(num_qubits=2, default=True)
        features = np.array([0.123])
        params = np.zeros(circuit.num_parameters, dtype=float)

        qc = circuit.get_circuit(features=features, parameters=params)

        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        # circuit should contain some gates (count_ops gives a dict of operations)
        ops = qc.count_ops()
        total_ops = sum(ops.values()) if ops else 0
        assert total_ops > 0, "Returned circuit appears empty (no gates found)."

    def test_set_params(self):
        circuit = QCNNEncodingCircuit(num_qubits=2, default=False)

        # add a convolutional layer
        circuit.convolution()
        assert len(circuit.operations_list) == 1
        old_num_parameters = circuit.num_parameters
        assert old_num_parameters >= 0

        # increase number of qubits
        circuit.set_params(num_qubits=4)
        assert circuit.num_qubits == 4
        assert circuit.left_qubits == [0, 1, 2, 3]  # left_qubits initially reset to the full range
        # after re-applying the stored Conv operation, num_parameters should be recomputed (>=0)
        assert isinstance(circuit.num_parameters, int)
        assert circuit.num_parameters >= 0

    def test_set_params_default(self):
        circuit = QCNNEncodingCircuit(num_qubits=2, default=True)
        assert circuit._default is True
        # ensure there is an initial default structure
        assert len(circuit.operations_list) >= 1

        # change number of qubits and ensure default circuit is rebuilt
        circuit.set_params(num_qubits=4)
        assert circuit.num_qubits == 4
        # default_circuit should create operations and end with FC
        assert len(circuit.operations_list) >= 1
        assert circuit.operations_list[-1][0] == "FC"

    def test_repeat_layers(self):
        circuit = QCNNEncodingCircuit(num_qubits=4, default=False)

        # add a default conv + pool sequence
        circuit.convolution()
        circuit.pooling()
        initial_ops_len = len(circuit.operations_list)
        initial_left_len = len(circuit.left_qubits)

        # repeat once
        circuit.repeat_layers(n_times=1)

        # operations_list should have grown (the earlier operations were reapplied)
        assert len(circuit.operations_list) == 2 * initial_ops_len

        # left_qubits should be equal or reduced (pooling reduces available qubits)
        assert len(circuit.left_qubits) <= initial_left_len

    def test_repeat_layers_raises_value_error(self):
        with pytest.raises(ValueError):
            QCNNEncodingCircuit(num_qubits=4, default=False).repeat_layers(n_times=-1)
        with pytest.raises(ValueError):
            QCNNEncodingCircuit(num_qubits=1, default=False).repeat_layers(n_times=1)

    def test_build_circuit(self):
        circuit = QCNNEncodingCircuit(num_qubits=0, default=False)
        pool_qc = QuantumCircuit(2)

        input_list = [[0, 1], [2, 3]]
        output_list = [[0], [2]]
        circuit._operations_list = [["Pool", pool_qc, "Pool", False, input_list, output_list]]

        circuit.build_circuit(final_num_qubits=2)
        assert circuit.num_qubits == 4

    def test_build_circuit_empty_output_list(self):
        circuit = QCNNEncodingCircuit(num_qubits=0, default=False)
        pool_qc = QuantumCircuit(2)

        circuit._operations_list = [["Pool", pool_qc, "Pool", False, [], []]]

        circuit.build_circuit(final_num_qubits=1)
        assert circuit.num_qubits == 2

    def test_build_circuit_raises_when_output_list_larger_than_final(self):
        circuit = QCNNEncodingCircuit(num_qubits=0, default=False)
        pool_qc = QuantumCircuit(2)

        input_list = [[0, 1], [2, 3]]
        output_list = [[0], [2, 3]]

        circuit._operations_list = [["Pool", pool_qc, "Pool", False, input_list, output_list]]

        with pytest.raises(ValueError):
            circuit.build_circuit(final_num_qubits=1)

    def test_qcnn_observable(self):
        circuit = QCNNEncodingCircuit(num_qubits=3, default=False)

        obs = circuit.QCNNObservable("Z")
        assert isinstance(obs, ObservableBase)
        assert obs.num_qubits == circuit.num_qubits

        # invalid string should raise ValueError
        with pytest.raises(ValueError):
            circuit.QCNNObservable("A")
