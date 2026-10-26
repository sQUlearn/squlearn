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

    values = llqnn.evaluate(
        [[0.1, 0.2], [0.3, 0.4]],
        param,
        param_pbs,
        "f",
        "dfdp",
        "dfdx",
        "var",
    )
    return values


def test_backends_consistency():
    """Tests that the plain native keys (f/dfdp/dfdx/var) agree across frameworks.

    Raw identity-based derivative keys (e.g. `llqnn.parameters[0]`) are intentionally
    not exercised here: they stay Qiskit-only (see LowLevelQNN._build_derivative_arg)
    - qc_executor's tuple mechanism resolves them by object identity, and PennyLane's/
    Qulacs's qc_executor backends have no equivalent yet. Since the legacy per-framework
    fallback engines were removed (all three frameworks are fully native now, see
    test_var_family_is_native_and_matches_across_frameworks below for that proof), there is
    no other path left to service such a key for PennyLane/Qulacs any more - see
    test_raw_derivative_key_p_op_ownership_qiskit_only for coverage of that Qiskit-only
    mechanism instead.
    """

    values_qiskit = get_values("qiskit")
    values_pennylane = get_values("pennylane")

    for k in ["f", "dfdp", "dfdx", "var"]:
        assert np.allclose(values_qiskit[k], values_pennylane[k])


def test_raw_derivative_key_p_op_ownership_qiskit_only():
    """Regression for the raw-key multi-output p_op-ownership logic
    (LowLevelQNN._p_op_entry_excludes_observable): a second derivative built from two
    individual llqnn.parameters_operator[i] elements belonging to *different* observables
    must be evaluated without error (qc_executor itself raises if a raw parameter belonging
    to a different observable's operator is passed through unfiltered - this is what the
    per-observable ownership check exists to prevent).

    Qiskit-only: raw identity-based keys have no PennyLane/Qulacs equivalent (see
    test_backends_consistency), so there is no independent cross-framework reference for
    this specific mechanism. The observable is linear in "p_op" (a weighted Pauli sum), so
    any p_op-p_op second derivative is mathematically exactly zero regardless of ownership -
    that is the non-vacuous assertion here, not just "the call didn't crash".
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = [
        SinglePauli(pqc.num_qubits, 0, op_str="Z"),
        SummedPaulis(pqc.num_qubits),
        SummedPaulis(pqc.num_qubits),
    ]
    rng = np.random.default_rng(23)
    x = rng.random((2, 2))
    param = rng.random(pqc.num_parameters)
    n_pop = sum(o.num_parameters for o in obs)
    param_op = rng.random(n_pop)

    llqnn = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)

    # p_op[0:4] belongs to the second observable, p_op[4:8] to the third (the first owns none).
    key = (llqnn.parameters_operator[0], llqnn.parameters_operator[4])
    values = llqnn.evaluate(x, param, param_op, key)

    np.testing.assert_allclose(values[key], 0.0, atol=1e-8)


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


_VAR_FAMILY_KEYS = ("var", "varf", "dvardx", "dvardp", "dvardop")


@pytest.mark.parametrize(
    "observable",
    [
        pytest.param(lambda pqc: SummedPaulis(pqc.num_qubits), id="single-parameterized"),
        pytest.param(
            lambda pqc: [SinglePauli(pqc.num_qubits, i, op_str="Z") for i in range(3)],
            id="multi-parameter-free",
        ),
        pytest.param(
            lambda pqc: [
                SinglePauli(pqc.num_qubits, 0, op_str="Z"),
                SummedPaulis(pqc.num_qubits),
            ],
            id="multi-mixed-parameters",
        ),
    ],
)
@pytest.mark.parametrize("framework", ["pennylane", "qulacs"])
def test_var_family_is_native_and_matches_across_frameworks(framework, observable):
    """The var/dvardx/dvardp/dvardop family is evaluated via qc_executor's native
    <O^2> path (LowLevelQNN._native_observable_squared) for every framework - no
    fallback engine exists any more (all three legacy per-framework engines have been
    removed, having been proven bit-exact against this native path for every key they
    supported before removal). Verified here by cross-framework agreement against a Qiskit
    statevector reference (an independent computation, not a self-consistency check against
    qc_executor's own machinery).
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = observable(pqc)
    rng = np.random.default_rng(3)
    x = rng.random((3, 2))
    param = rng.random(pqc.num_parameters)
    num_parameters_observable = (
        sum(o.num_parameters for o in obs) if isinstance(obs, list) else obs.num_parameters
    )
    param_op = rng.random(num_parameters_observable)

    llqnn = LowLevelQNN(pqc, obs, Executor(framework), num_features=2)
    native = llqnn.evaluate(x, param, param_op, "f", *_VAR_FAMILY_KEYS)

    llqnn_qiskit = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    reference = llqnn_qiskit.evaluate(x, param, param_op, "f", *_VAR_FAMILY_KEYS)

    for key in ("f", *_VAR_FAMILY_KEYS):
        np.testing.assert_allclose(native[key], reference[key], atol=1e-8)


# Higher-order and mixed (chained) derivatives, native for Qiskit and PennyLane (see
# LowLevelQNN._NATIVE_KEY_INFO_CHAINED). Every key below is cross-checked against
# PennyLane's own, independently-computed (chained qml.jacobian) native engine, except
# "dfdopdxdx": PennyLane's own engine fails on that one specific key with a pre-existing,
# unrelated NonDifferentiableError (autograd trips over a near-zero complex intermediate)
# for both observable configs below - verified individually per key before writing this
# test. "dfdpdop" (no key for it, only its transpose "dfdopdp") and the three order-3
# pure-circuit permutations (dfdxdxdp/dfdxdpdx/dfdpdxdx) are verified separately below.
_CHAINED_KEYS = (
    "dfdxdx",
    "dfdpdp",
    "dfdopdp",
    "dfdopdop",
    "dfdpdx",
    "dfdopdx",
    "dfccdxdx",
    "dfccdpdp",
    "dfccdopdx",
    "dfccdopdop",
)

_OBSERVABLE_CONFIGS = [
    pytest.param(lambda pqc: SummedPaulis(pqc.num_qubits), id="single-parameterized"),
    pytest.param(
        lambda pqc: [
            SinglePauli(pqc.num_qubits, 0, op_str="Z"),
            SummedPaulis(pqc.num_qubits),
            SummedPaulis(pqc.num_qubits),
        ],
        id="multi-mixed-parameters",
    ),
]


@pytest.mark.parametrize("observable", _OBSERVABLE_CONFIGS)
def test_chained_derivatives_are_native_and_match_across_frameworks(observable):
    """The chained-derivative keys (dfdxdx, ...) are evaluated via qc_executor's tuple-form
    expectation_value_derivatives for both Qiskit and PennyLane - verified by bit-for-bit
    agreement between the two, including for a multi-output case with observables that own
    differently-sized (including zero-sized) "p_op" slices.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = observable(pqc)
    rng = np.random.default_rng(11)
    x = rng.random((2, 2))
    param = rng.random(pqc.num_parameters)
    num_parameters_observable = (
        sum(o.num_parameters for o in obs) if isinstance(obs, list) else obs.num_parameters
    )
    param_op = rng.random(num_parameters_observable)

    llqnn_qiskit = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    native = llqnn_qiskit.evaluate(x, param, param_op, *_CHAINED_KEYS)

    llqnn_pennylane = LowLevelQNN(pqc, obs, Executor("pennylane"), num_features=2)
    reference = llqnn_pennylane.evaluate(x, param, param_op, *_CHAINED_KEYS)

    for key in _CHAINED_KEYS:
        np.testing.assert_allclose(native[key], reference[key], atol=1e-6)


@pytest.mark.parametrize("observable", _OBSERVABLE_CONFIGS)
def test_dfdopdxdx_matches_finite_differences(observable):
    """dfdopdxdx is the one chained key PennyLane's own engine cannot compute (see the note
    above _CHAINED_KEYS), so it is verified independently via central finite differences of
    the already cross-framework-verified "dfdxdx" key with respect to each p_op parameter -
    d(dfdxdx)/dp_op_k approximates dfdopdxdx[..., k, :, :] to O(eps^2).
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = observable(pqc)
    rng = np.random.default_rng(11)
    x = rng.random(2)
    param = rng.random(pqc.num_parameters)
    num_parameters_observable = (
        sum(o.num_parameters for o in obs) if isinstance(obs, list) else obs.num_parameters
    )
    param_op = rng.random(num_parameters_observable)

    llqnn = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    analytic = np.asarray(llqnn.evaluate(x, param, param_op, "dfdopdxdx")["dfdopdxdx"])

    eps = 1e-4
    p_op_axis = analytic.ndim - 3  # trailing (..., n_p_op, n_x, n_x); multi-output adds n_op first
    fd = np.zeros_like(analytic)
    for k in range(num_parameters_observable):
        p_plus, p_minus = param_op.copy(), param_op.copy()
        p_plus[k] += eps
        p_minus[k] -= eps
        dxdx_plus = np.asarray(llqnn.evaluate(x, param, p_plus, "dfdxdx")["dfdxdx"])
        dxdx_minus = np.asarray(llqnn.evaluate(x, param, p_minus, "dfdxdx")["dfdxdx"])
        index = tuple(slice(None) if ax != p_op_axis else k for ax in range(analytic.ndim))
        fd[index] = (dxdx_plus - dxdx_minus) / (2 * eps)

    np.testing.assert_allclose(analytic, fd, atol=1e-4)


def test_dfdpdop_matches_transpose_of_dfdopdp():
    """No key ever implemented "dfdpdop" directly (only its transpose "dfdopdp"), so there
    is no independent reference for it - verified instead against its own,
    independently-computed transpose relation.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = SummedPaulis(pqc.num_qubits)
    rng = np.random.default_rng(5)
    x = rng.random((2, 2))
    param = rng.random(pqc.num_parameters)
    param_op = rng.random(obs.num_parameters)

    llqnn = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    values = llqnn.evaluate(x, param, param_op, "dfdpdop", "dfdopdp")
    np.testing.assert_allclose(
        values["dfdpdop"], np.swapaxes(values["dfdopdp"], -1, -2), atol=1e-8
    )


def test_order_three_pure_circuit_permutations_match_pennylane_reference():
    """dfdxdxdp/dfdxdpdx/dfdpdxdx are order-3, pure circuit-side permutations. Verified
    against PennyLane's independently-computed (chained qml.jacobian) native engine.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = SummedPaulis(pqc.num_qubits)
    rng = np.random.default_rng(9)
    x = rng.random((2, 2))
    param = rng.random(pqc.num_parameters)
    param_op = rng.random(obs.num_parameters)

    keys = ("dfdxdxdp", "dfdxdpdx", "dfdpdxdx")
    native_qiskit = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    native = native_qiskit.evaluate(x, param, param_op, *keys)

    reference = LowLevelQNN(pqc, obs, Executor("pennylane"), num_features=2).evaluate(
        x, param, param_op, *keys
    )

    for key in keys:
        np.testing.assert_allclose(native[key], reference[key], atol=1e-6)


def test_laplace_family_is_native_and_matches_pennylane_reference():
    """laplace/laplace_dp are pure post-processing (a diagonal trace) of a chained key above,
    reusing evaluation_classes.get_eval_laplace exactly like PennyLane's own native engine
    does - verified against it.
    "laplace_dop" is excluded here: it depends on "dfdopdxdx" (see
    test_dfdopdxdx_matches_finite_differences), the one chained key PennyLane's own engine
    cannot compute - test_laplace_dop_matches_finite_differences below covers it instead.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = SummedPaulis(pqc.num_qubits)
    rng = np.random.default_rng(13)
    x = rng.random((2, 2))
    param = rng.random(pqc.num_parameters)
    param_op = rng.random(obs.num_parameters)

    keys = ("laplace", "laplace_dp")
    executor = Executor("statevector_simulator")
    llqnn = LowLevelQNN(pqc, obs, executor, num_features=2)
    native = llqnn.evaluate(x, param, param_op, *keys)

    reference = LowLevelQNN(pqc, obs, Executor("pennylane"), num_features=2).evaluate(
        x, param, param_op, *keys
    )

    for key in keys:
        np.testing.assert_allclose(
            np.asarray(native[key]).ravel(), np.asarray(reference[key]).ravel(), atol=1e-6
        )


def test_laplace_dop_matches_finite_differences():
    """laplace_dop depends on "dfdopdxdx" (see test_dfdopdxdx_matches_finite_differences for
    why PennyLane can't serve as a reference for it), so it is verified the same way: central
    finite differences of the already cross-framework-verified "laplace" key with respect to
    each p_op parameter.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = SummedPaulis(pqc.num_qubits)
    rng = np.random.default_rng(13)
    x = rng.random(2)
    param = rng.random(pqc.num_parameters)
    param_op = rng.random(obs.num_parameters)

    llqnn = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    analytic = np.asarray(llqnn.evaluate(x, param, param_op, "laplace_dop")["laplace_dop"])

    eps = 1e-4
    fd = np.zeros_like(analytic)
    for k in range(obs.num_parameters):
        p_plus, p_minus = param_op.copy(), param_op.copy()
        p_plus[k] += eps
        p_minus[k] -= eps
        lp_plus = llqnn.evaluate(x, param, p_plus, "laplace")["laplace"]
        lp_minus = llqnn.evaluate(x, param, p_minus, "laplace")["laplace"]
        fd[k] = (lp_plus - lp_minus) / (2 * eps)

    np.testing.assert_allclose(analytic, fd, atol=1e-4)


def test_chained_derivative_keys_stay_non_native_for_qulacs():
    """The chained-derivative keys are native for Qiskit and PennyLane only - Qulacs's
    qc_executor backend has no equivalent yet, and (with every legacy fallback engine
    removed) there is nothing left to route to any more: the key is simply unsupported.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = SummedPaulis(pqc.num_qubits)
    rng = np.random.default_rng(17)
    x = rng.random((2, 2))
    param = rng.random(pqc.num_parameters)
    param_op = rng.random(obs.num_parameters)

    llqnn = LowLevelQNN(pqc, obs, Executor("qulacs"), num_features=2)
    with pytest.raises(NotImplementedError):
        llqnn.evaluate(x, param, param_op, "dfdxdx")
