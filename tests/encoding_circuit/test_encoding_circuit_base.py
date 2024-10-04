from squlearn.encoding_circuit.circuit_library.chebyshev_pqc import ChebyshevPQC
from squlearn.encoding_circuit.encoding_circuit_base import (
    EncodingCircuitBase,
    EncodingSlotsMismatchError,
)
from qiskit.circuit import ParameterVector
import pytest


class TestEncodingCircuitBase:
    """Test class for EncodingCircuitBase"""

    def test_encoding_slots_mismatch(self):
        fm = EncodingCircuitBase(num_qubits=2, num_features=3)
        with pytest.raises(EncodingSlotsMismatchError):
            fm.get_circuit(features=ParameterVector("x", 3), parameters=ParameterVector("p", 2))
