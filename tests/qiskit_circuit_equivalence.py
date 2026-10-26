import numpy as np
from qc_executor import QuantumCircuit


def assert_circuits_equal(
    qc_actual: QuantumCircuit, qc_expected: QuantumCircuit, tol: float = 1e-8
):
    """Asserts that two Qiskit QuantumCircuits are equivalent in terms of their operations, qubits, and parameters."""

    assert qc_actual.num_qubits == qc_expected.num_qubits, "different number of qubits"
    assert len(qc_actual.qiskit_circuit.data) == len(
        qc_expected.qiskit_circuit.data
    ), f"different number of instructions: {len(qc_actual.qiskit_circuit.data)} vs {len(qc_expected.qiskit_circuit.data)}"

    for i, (inst_act, inst_exp) in enumerate(
        zip(qc_actual.qiskit_circuit.data, qc_expected.qiskit_circuit.data)
    ):
        op_act = inst_act.operation
        op_exp = inst_exp.operation

        qargs_act = tuple(qc_actual.qiskit_circuit.qubits.index(q) for q in inst_act.qubits)
        qargs_exp = tuple(qc_expected.qiskit_circuit.qubits.index(q) for q in inst_exp.qubits)
        assert (
            qargs_act == qargs_exp
        ), f"Instruction {i}: qubit targets differ: {qargs_act} != {qargs_exp}"

        def params_to_array(params):
            return np.array([float(p) for p in params], dtype=float)

        p_act = params_to_array(op_act.params)
        p_exp = params_to_array(op_exp.params)
        assert (
            p_act.shape == p_exp.shape
        ), f"Instruction {i}: param shapes differ: {p_act.shape} != {p_exp.shape}"
        if p_act.size > 0:
            assert np.allclose(
                p_act, p_exp, atol=tol, rtol=0
            ), f"Instruction {i}: params differ: {p_act} != {p_exp}"
