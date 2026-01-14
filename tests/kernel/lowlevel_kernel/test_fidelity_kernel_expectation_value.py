import numpy as np
import pytest
from unittest.mock import MagicMock

# Import the module under test (we will monkeypatch LowLevelQNN inside this module)
import squlearn.kernel.lowlevel_kernel.fidelity_kernel_expectation_value as fk_mod
from squlearn.kernel.lowlevel_kernel.fidelity_kernel_expectation_value import (
    FidelityKernelExpectationValue,
)


class FakeLowLevelQNN:
    """
    Minimal fake of LowLevelQNN used for tests.

    The real LowLevelQNN.evaluate(...) returns a mapping from todo-string to numpy array.
    Our fake returns deterministic arrays depending on 'todo' and the number of pairs nf.
    """

    def __init__(self, circuit, observable, executor=None, num_features=None):
        # store info that tests may inspect
        self.circuit = circuit
        self.observable = observable
        self.executor = executor
        self.num_features = num_features
        self.num_parameters = getattr(self, "num_parameters", 1)

    def evaluate(self, x, param, param_op, todo):
        """
        x: array shape (nf, 2*m) where nf = number of pairs
        param / param_op: passed through but ignored in the fake
        todo: string like "f", "dfdp", "dfdxdx" etc.
        Must return dict with todo -> numpy array
        """
        # nf = number of flattened pair rows
        nf = x.shape[0]

        if todo == "f":
            # Return a 1D array of length nf with deterministic values
            return {"f": np.array([0.11 * (i + 1) for i in range(nf)])}
        if todo == "dfdp":
            # produce array (nf, num_parameters) with values 1..(nf*num_parameters)
            num_p = getattr(self, "num_parameters", 1)
            arr = np.arange(1, nf * num_p + 1.0, dtype=float).reshape(nf, num_p)
            return {"dfdp": arr}
        if todo == "dfdxdx":
            # return something with the correct flattened size for jacobian path (not used deeply)
            return {"dfdxdx": np.zeros((nf, (2 * (self.num_features or 1)) ** 2))}
        # default: zeros
        return {todo: np.zeros(nf)}


def make_encoding_circuit_mock(num_qubits=1):
    enc = MagicMock()
    enc.num_qubits = num_qubits
    enc.compose = MagicMock(return_value="composed_circuit")
    enc.inverse = MagicMock(return_value="inverse_circuit")
    enc.get_params = MagicMock(return_value={})
    enc.set_params = MagicMock()
    enc.num_parameters = 0
    return enc


def make_executor_mock():
    exec_mock = MagicMock()
    exec_mock.shots = None
    exec_mock.set_shots = MagicMock()
    return exec_mock


class TestFidelityKernelExpectationValue:

    @pytest.fixture(autouse=True)
    def _patch_lowlevelqnn(self, monkeypatch):
        """
        Patch the LowLevelQNN symbol in the module under test so that when
        FidelityKernelExpectationValue constructs a LowLevelQNN it gets our fake.
        """
        monkeypatch.setattr(fk_mod, "LowLevelQNN", FakeLowLevelQNN)
        fk_mod.FakeLowLevelQNN = FakeLowLevelQNN
        yield

    def test_init(self):
        circuit = make_encoding_circuit_mock(num_qubits=2)
        executor = make_executor_mock()
        kernel = FidelityKernelExpectationValue(circuit, executor, "all", True)
        assert kernel.encoding_circuit.num_qubits == 2
        assert kernel._caching is True
        assert kernel._evaluate_duplicates == "all"
        assert kernel.parameters is None
        assert kernel._derivative_cache == {}
        assert kernel._qnn is None
        assert kernel._executor is not None

    def test_assign_training_parameters(self):
        circuit = make_encoding_circuit_mock(num_qubits=2)
        executor = make_executor_mock()
        kernel = FidelityKernelExpectationValue(circuit, executor, "all", True)
        kernel.assign_training_parameters(np.array([0.1, 0.2, 0.3, 0.4]))
        assert np.array_equal(kernel.parameters, np.array([0.1, 0.2, 0.3, 0.4]))

    def test_evaluate_returns_kernel_matrix_for_all_duplicates(self):
        x = np.array([[0.1], [0.2]])
        fk = FidelityKernelExpectationValue(
            encoding_circuit=make_encoding_circuit_mock(num_qubits=1),
            executor=make_executor_mock(),
            evaluate_duplicates="all",
            caching=False,
        )

        fk.encoding_circuit.num_parameters = 0
        fk._parameters = None

        K = fk.evaluate(x)

        # FakeLowLevelQNN returns f = [0.11, 0.22, 0.33, 0.44] for nf=4
        expected = np.array([[0.11, 0.22], [0.33, 0.44]])
        assert K.shape == (2, 2)
        assert np.allclose(K, expected)

    def test_evaluate_derivatives_dKdp_shape_and_values(self):
        # ensure dKdp (num_parameters, n, n) shaping is correct
        x = np.array([[0.1], [0.2]])
        fk = FidelityKernelExpectationValue(
            encoding_circuit=make_encoding_circuit_mock(num_qubits=1),
            executor=make_executor_mock(),
            evaluate_duplicates="all",
            caching=False,
        )

        # set number of parameters expected
        fk.encoding_circuit.num_parameters = 3
        fk._parameters = []  # allow code to proceed without raising

        # make FakeLowLevelQNN produce arrays consistent with fk.num_parameters
        FakeLowLevelQNN.num_parameters = fk.num_parameters
        FakeLowLevelQNN.num_features = x.shape[1]

        out = fk.evaluate_derivatives(x, x, values="dKdp")

        assert isinstance(out, np.ndarray)
        assert out.shape == (fk.num_parameters, x.shape[0], x.shape[0])

        nf = x.shape[0] * x.shape[0]
        full = np.arange(1.0, nf * fk.num_parameters + 1.0).reshape(nf, fk.num_parameters)
        reshaped = full.reshape(x.shape[0], x.shape[0], fk.num_parameters)
        expected = reshaped.transpose(2, 0, 1)
        assert_all = np.allclose(out, expected)
        assert assert_all

    def test_evaluate_derivatives_single_value_and_caching_behaviour(self):
        """Requesting a single string value returns an array (not a dict), and caching works."""
        x = np.array([[0.1], [0.2]])
        fk = FidelityKernelExpectationValue(
            encoding_circuit=make_encoding_circuit_mock(num_qubits=1),
            executor=make_executor_mock(),
            evaluate_duplicates="all",
            caching=True,
        )

        fk.encoding_circuit.num_parameters = 0
        fk._parameters = None

        # First call should compute and store in cache
        first = fk.evaluate_derivatives(x, x, values="K")
        assert isinstance(first, np.ndarray)
        assert first.shape == (2, 2)

        # call again with same args: should use cache and return same result
        second = fk.evaluate_derivatives(x, x, values="K")
        assert np.allclose(first, second)
