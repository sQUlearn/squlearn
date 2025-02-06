import copy
from squlearn.encoding_circuit.circuit_library.chebyshev_pqc import ChebyshevPQC
from squlearn.encoding_circuit.transpiled_encoding_circuit import TranspiledEncodingCircuit
from qiskit_ibm_runtime.fake_provider import FakeManilaV2


class TestTranspiledEncodingCircuit:

    def test_drawing_does_not_violate_circuit_parameters(self):
        circuit = ChebyshevPQC(num_features=2, num_qubits=4)
        transpiled_circuit = TranspiledEncodingCircuit(
            encoding_circuit=circuit, backend=FakeManilaV2()
        )

        params_with_features_before = copy.deepcopy(transpiled_circuit.get_params())
        qubit_map_before = copy.copy(transpiled_circuit.qubit_map)
        circuit.draw(output="mpl")
        params_with_features_after = copy.deepcopy(transpiled_circuit.get_params())
        qubit_map_after = copy.copy(transpiled_circuit.qubit_map)

        assert params_with_features_before == params_with_features_after
        assert qubit_map_before == qubit_map_after

        # same but with num_features=None
        circuit = ChebyshevPQC(num_features=None, num_qubits=4)
        transpiled_circuit = TranspiledEncodingCircuit(
            encoding_circuit=circuit, backend=FakeManilaV2()
        )

        params_without_features_before = copy.deepcopy(transpiled_circuit.get_params())
        qubit_map_before = copy.deepcopy(transpiled_circuit.qubit_map)
        circuit.draw(output="mpl")
        params_without_features_after = copy.deepcopy(transpiled_circuit.get_params())
        qubit_map_after = copy.deepcopy(transpiled_circuit.qubit_map)

        assert params_without_features_before == params_without_features_after
        assert qubit_map_before == qubit_map_after
