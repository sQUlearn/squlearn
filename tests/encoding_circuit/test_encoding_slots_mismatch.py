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


@pytest.mark.parametrize(
    "circuit_class, init_params",
    [
        (ChebyshevPQC, {"num_qubits": 2, "num_layers": 1, "num_features": 3}),
        (ChebyshevRx, {"num_qubits": 2, "num_layers": 1, "num_features": 3}),
        (ChebyshevTower, {"num_qubits": 2, "num_chebyshev": 1, "num_features": 3}),
        (HubregtsenEncodingCircuit, {"num_qubits": 2, "num_layers": 1, "num_features": 3}),
        (KyriienkoEncodingCircuit, {"num_qubits": 2, "num_features": 3}),
        (MultiControlEncodingCircuit, {"num_qubits": 2, "num_layers": 1, "num_features": 3}),
        (QCNNEncodingCircuit, {"num_qubits": 2, "num_features": 3}),
        (YZ_CX_EncodingCircuit, {"num_qubits": 2, "num_layers": 1, "num_features": 3}),
    ],
)
class TestEncodingSlotsMismatchError:

    def test_encoding_slots_mismatch_error(self, circuit_class, init_params):
        """
        Test that each circuit raises EncodingSlotsMismatchError
        when the number of features does not match the number of encoding slots.
        """
        circuit = circuit_class(**init_params)
        with pytest.raises(EncodingSlotsMismatchError):
            circuit.get_circuit([], [])
