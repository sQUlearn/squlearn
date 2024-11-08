import pytest
from squlearn.encoding_circuit import (
    ChebyshevPQC,
    ChebyshevRx,
    ChebyshevTower,
    HubregtsenEncodingCircuit,
    KyriienkoEncodingCircuit,
    MultiControlEncodingCircuit,
    QCNNEncodingCircuit,
    YZ_CX_EncodingCircuit,
)
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError


class TestEncodingSlotsMismatchError:
    def test_chebyshev_pqc(self):
        circuit = ChebyshevPQC(num_qubits=2, num_layers=1, num_features=3)
        with pytest.raises(EncodingSlotsMismatchError):
            circuit.get_circuit([], [])

    def test_chebyshev_rx(self):
        circuit = ChebyshevRx(num_qubits=2, num_layers=1, num_features=3)
        with pytest.raises(EncodingSlotsMismatchError):
            circuit.get_circuit([], [])

    def test_chebyshev_tower(self):
        circuit = ChebyshevTower(num_qubits=2, num_chebyshev=1, num_features=3)
        with pytest.raises(EncodingSlotsMismatchError):
            circuit.get_circuit([], [])

    def test_hubregtsen_encoding_circuit(self):
        circuit = HubregtsenEncodingCircuit(num_qubits=2, num_layers=1, num_features=3)
        with pytest.raises(EncodingSlotsMismatchError):
            circuit.get_circuit([], [])

    def test_kyriienko_encoding_circuit(self):
        circuit = KyriienkoEncodingCircuit(num_qubits=2, num_features=3)
        with pytest.raises(EncodingSlotsMismatchError):
            circuit.get_circuit([], [])

    def test_multi_control_encoding_circuit(self):
        circuit = MultiControlEncodingCircuit(num_qubits=2, num_layers=1, num_features=3)
        with pytest.raises(EncodingSlotsMismatchError):
            circuit.get_circuit([], [])

    def test_qcnn_encoding_circuit(self):
        circuit = QCNNEncodingCircuit(num_qubits=2, num_features=3)
        with pytest.raises(EncodingSlotsMismatchError):
            circuit.get_circuit([], [])

    def test_yz_cx_encoding_circuit(self):
        circuit = YZ_CX_EncodingCircuit(num_qubits=2, num_layers=1, num_features=3)
        with pytest.raises(EncodingSlotsMismatchError):
            circuit.get_circuit([], [])
