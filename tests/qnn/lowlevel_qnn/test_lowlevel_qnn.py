import pytest
import numpy as np


from squlearn import Executor
from squlearn.encoding_circuit import ParamZFeatureMap
from squlearn.observables import SinglePauli, SummedPaulis
from squlearn.qnn.lowlevel_qnn import LowLevelQNN


def get_values(framework):
    executor = Executor(framework)
    pqc = ParamZFeatureMap(4, 2)
    obs1 = SummedPaulis(4)
    obs2 = SummedPaulis(4)

    llqnn = LowLevelQNN(pqc, [obs1, obs2], executor=executor, num_features=2)

    np.random.seed(42)
    param = np.random.rand(2, llqnn.num_parameters)
    param_pbs = np.random.rand(2, llqnn.num_parameters_observable)

    return llqnn.evaluate(
        [[0.1, 0.2], [0.3, 0.4]],
        param,
        param_pbs,
        "f",
        "dfdp",
        "dfdx",
        "var",
        (llqnn.parameters[0],),
        (llqnn.parameters_operator[0],),
    )


def test_backends_consistency():
    """Tests that different derivatives computed with different frameworks are consistent."""

    values_qiskit = get_values("qiskit")
    values_pennylane = get_values("pennylane")
    values_qulacs = get_values("qulacs")

    for k in ["f", "dfdp", "dfdx", "var"]:
        assert np.allclose(values_qiskit[k], values_pennylane[k])
        assert np.allclose(values_qiskit[k], values_qulacs[k])

    qiskit_keys = sorted(list(values_qiskit.keys()), key=str)
    pennylane_keys = sorted(list(values_pennylane.keys()), key=str)
    qulacs_keys = sorted(list(values_qulacs.keys()), key=str)

    qiskit_param_key = qiskit_keys[0]
    pennylane_param_key = pennylane_keys[2]
    qulacs_param_key = qulacs_keys[2]

    qiskit_param_op_key = qiskit_keys[1]
    pennylane_param_op_key = pennylane_keys[3]
    qulacs_param_op_key = qulacs_keys[3]

    assert np.allclose(values_qiskit[qiskit_param_key], values_pennylane[pennylane_param_key])
    assert np.allclose(values_qiskit[qiskit_param_key], values_qulacs[qulacs_param_key])

    assert np.allclose(
        values_qiskit[qiskit_param_op_key], values_pennylane[pennylane_param_op_key]
    )
    assert np.allclose(values_qiskit[qiskit_param_op_key], values_qulacs[qulacs_param_op_key])


@pytest.mark.parametrize("framework", ["pennylane", "qiskit", "qulacs"])
@pytest.mark.parametrize("n_obs", [1, 2])
def test_multiple_output_shape_with_n_observables(framework, n_obs):
    """Regression: a list-of-one observable must produce the same shape contract
    as a list of multiple observables. PennyLane otherwise collapses the leading
    "observable" axis when only one measurement is returned, which previously
    crashed ProjectedQuantumKernel for num_qubits=1 with a single-Pauli measurement.
    """
    pqc = ParamZFeatureMap(4, 2)
    obs = [SinglePauli(4, i, op_str="X") for i in range(n_obs)]
    llqnn = LowLevelQNN(pqc, obs, executor=Executor(framework), num_features=2)

    np.random.seed(42)
    x = np.random.rand(5, 2)
    param = np.random.rand(llqnn.num_parameters)

    assert llqnn.evaluate(x, param, [], "f")["f"].shape == (5, n_obs)
    assert llqnn.evaluate(x, param, [], "dfdx")["dfdx"].shape == (5, n_obs, 2)
    assert llqnn.evaluate(x, param, [], "dfdp")["dfdp"].shape == (
        5,
        n_obs,
        llqnn.num_parameters,
    )
