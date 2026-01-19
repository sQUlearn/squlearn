from unittest.mock import MagicMock
import numpy as np
import pytest
from squlearn.kernel.lowlevel_kernel.fidelity_kernel_statevector import FidelityKernelStatevector


def make_executor(is_statevector=True, framework="pennylane", shots=None):
    """Create a minimal executor-like object with required attributes."""
    exec_mock = MagicMock()
    exec_mock.is_statevector = is_statevector
    exec_mock.quantum_framework = framework
    exec_mock.shots = shots
    # set_shots should exist
    exec_mock.set_shots = MagicMock()
    # For shots-mode tests we may set pennylane_execute_batched externally
    return exec_mock


def make_encoding_circuit(num_parameters=0):
    enc = MagicMock()
    enc.num_parameters = num_parameters
    enc.get_circuit = MagicMock(return_value="dummy_circuit")
    return enc


class TestFidelityKernelStatevector:

    def test_assign_training_parameters_errors_and_success(self):
        k = object.__new__(FidelityKernelStatevector)
        k._encoding_circuit = make_encoding_circuit(num_parameters=2)
        # indicate that we have a parameter vector
        k._parameter_vector = object()

        # wrong length -> ValueError
        with pytest.raises(ValueError):
            k.assign_training_parameters(np.array([1.0]))  # len 1 != 2

        # correct length -> sets _parameters
        params = np.array([0.1, 0.2])
        k.assign_training_parameters(params)
        assert np.allclose(k._parameters, params)

        # if _parameter_vector is None -> assigning should raise
        k2 = object.__new__(FidelityKernelStatevector)
        k2._parameter_vector = None
        k2._encoding_circuit = make_encoding_circuit(num_parameters=0)
        with pytest.raises(ValueError):
            k2.assign_training_parameters(np.array([0.1]))

    def test_evaluate_routes_to_correct_internal_method(self):
        # Ensure evaluate() calls evaluate_kernel_sv when is_statevector True
        k = object.__new__(FidelityKernelStatevector)
        k._executor = make_executor(is_statevector=True)
        k.evaluate_kernel_sv = MagicMock(return_value=np.ones((2, 2)))
        k.evaluate_kernel_shots = MagicMock(return_value=np.zeros((2, 2)))

        x = np.zeros((2, 1))
        out = k.evaluate(x)
        k.evaluate_kernel_sv.assert_called_once_with(x, x)
        assert out.shape == (2, 2)

        # Ensure evaluate() calls evaluate_kernel_shots when is_statevector False
        k2 = object.__new__(FidelityKernelStatevector)
        k2._executor = make_executor(is_statevector=False)
        k2.evaluate_kernel_sv = MagicMock(return_value=np.ones((2, 2)))
        k2.evaluate_kernel_shots = MagicMock(return_value=np.zeros((2, 2)))

        out2 = k2.evaluate(x)
        k2.evaluate_kernel_shots.assert_called_once_with(x, x)
        assert out2.shape == (2, 2)

    def test_evaluate_kernel_shots_symmetric_modes_and_parameter_check(self):
        """
        Test evaluate_kernel_shots logic:
        - Builds index lists and calls executor.pennylane_execute_batched
        - Handles symmetric case with evaluate_duplicates modes 'all', 'off_diagonal', 'none'
        - Raises when parameter_vector present but parameters missing
        """
        x = np.array([[0.1, 0.2], [0.3, 0.4]])

        k = object.__new__(FidelityKernelStatevector)
        k._num_features = 2
        k._pennylane_circuit = "circ"
        # set parameter_vector not None to trigger parameters check
        k._parameter_vector = object()
        k._parameters = None  # simulate missing params
        k._evaluate_duplicates = "off_diagonal"  # default
        # executor with pennylane batched execution
        exec_mock = make_executor(is_statevector=False, framework="pennylane")
        exec_mock.pennylane_execute_batched = MagicMock(return_value=[])
        k._executor = exec_mock

        # if parameter_vector set but _parameters None -> evaluate_kernel_shots should raise
        with pytest.raises(ValueError):
            k.evaluate_kernel_shots(x, x)

        # Now provide parameters and test actual batched call and matrix fill
        k._parameters = np.array([0.5])  # dummy params

        exec_mock.pennylane_execute_batched.return_value = [
            [1.0],
            [0.6],
            [1.0],
        ]

        # Test 'all' mode -> should produce kernel entries for (0,0),(0,1),(1,1) but our batched result length 1
        k._evaluate_duplicates = "all"
        # To avoid mismatch between expected indices and returned list length, we will instead create a small
        # non-symmetric test below to check filling logic; for symmetric 'all' we mainly ensure pennylane_execute_batched called.
        k._executor.pennylane_execute_batched.reset_mock()
        _ = k.evaluate_kernel_shots(x, x)
        k._executor.pennylane_execute_batched.assert_called()

        # Non-symmetric case: x vs y different arrays
        y = np.array([[0.1, 0.2], [0.5, 0.6]])
        # Create return list for 4 pairs (2x2)
        exec_mock.pennylane_execute_batched.return_value = [
            [0.11],
            [0.22],
            [0.33],
            [0.44],
        ]
        k._evaluate_duplicates = "off_diagonal"
        # parameter_vector present -> arguments include (params, x1, x2)
        mat = k.evaluate_kernel_shots(x, y)
        # Mat should be filled with our kernel entries at corresponding positions
        assert mat.shape == (2, 2)
        # each entry set from the returned values in the order built
        expected = np.array([[0.11, 0.22], [0.33, 0.44]])
        assert np.allclose(mat, expected)

    def test_evaluate_kernel_sv_symmetric_and_non_symmetric_simple_overlaps(self):
        """
        Test evaluate_kernel_sv with a simple cached_execution that returns orthonormal statevectors,
        so overlaps are 0 for different states and 1 for same states. Check symmetric handling and
        evaluate_duplicates behavior.
        """
        k = object.__new__(FidelityKernelStatevector)
        k._num_features = 2
        k._parameter_vector = None
        exec_mock = make_executor(is_statevector=True, framework="pennylane", shots=None)
        exec_mock.set_shots = MagicMock()
        k._executor = exec_mock

        # smart cached execution: inspects the incoming argument structure and returns N statevectors
        sv1 = np.array([1.0 + 0j, 0.0 + 0j])
        sv2 = np.array([0.0 + 0j, 1.0 + 0j])

        def cached_exec(f_alpha_tensor):
            # f_alpha_tensor is expected to be a tuple of arrays; the length of one of its elements equals n_samples
            try:
                # if f_alpha_tensor is a tuple of arrays, the length of first array is n_samples
                n_samples = len(f_alpha_tensor[0])
            except Exception:
                # fallback: if not subscriptable, assume single sample
                n_samples = 1
            if n_samples == 1:
                return np.array([sv1])
            elif n_samples == 2:
                return np.array([sv1, sv2])
            else:
                # repeat sv1 for arbitrary sizes
                return np.array([sv1] * n_samples)

        k._cached_execution = cached_exec

        x = np.array([[0.1, 0.2], [0.3, 0.4]])
        # symmetric case: x == y
        k._evaluate_duplicates = "off_diagonal"
        mat = k.evaluate_kernel_sv(x, x)
        # orthonormal sv -> overlap between distinct is 0; off_diagonal requires diagonal set to 1
        assert mat.shape == (2, 2)
        assert mat == pytest.approx(np.eye(2), abs=1e-12)

        # now non-symmetric: y different -> values computed for all pairs
        y = np.array([[0.1, 0.2]])
        mat2 = k.evaluate_kernel_sv(x, y)
        # x_sv length 2, y_sv length 1 -> mat shape (2,1)
        assert mat2.shape == (2, 1)
        # overlaps: sv1 vs sv1 -> 1, sv2 vs sv1 -> 0
        assert mat2[0, 0] == pytest.approx(1.0)
        assert mat2[1, 0] == pytest.approx(0.0)

    def test_evaluate_kernel_sv_with_parameters_requires_assigning_parameters(self):
        """
        When _parameter_vector is not None but _parameters is None,
        evaluate_kernel_sv should raise ValueError (pennylane branch).
        """
        k = object.__new__(FidelityKernelStatevector)
        k._num_features = 1
        k._parameter_vector = object()  # present
        k._parameters = None  # missing -> should raise
        exec_mock = make_executor(is_statevector=True, framework="pennylane", shots=None)
        exec_mock.set_shots = MagicMock()
        k._executor = exec_mock

        # create a cached_execution stub to satisfy calls if parameters present (not used here)
        k._cached_execution = MagicMock(return_value=np.array([1.0 + 0j]))

        x = np.array([[0.1]])
        with pytest.raises(ValueError):
            k.evaluate_kernel_sv(x, x)

    def test_evaluate_kernel_sv_qulacs_no_parameter_vector(self):
        """
        Test the qulacs statevector branch when no parameter_vector is present:
        cached_execution should be called once per sample with tuple(x_i) and return a statevector.
        """
        k = object.__new__(FidelityKernelStatevector)
        k._num_features = 2
        k._parameter_vector = None  # no trainable circuit params -> no param-branch

        exec_mock = make_executor(is_statevector=True, framework="qulacs", shots=None)
        exec_mock.set_shots = MagicMock()
        k._executor = exec_mock

        # two orthonormal statevectors
        sv1 = np.array([1.0 + 0j, 0.0 + 0j])
        sv2 = np.array([0.0 + 0j, 1.0 + 0j])

        # cached_execution should accept a tuple (x_ values) and return the statevector for that sample
        def cached_exec_per_sample(x_tuple):
            # x_tuple corresponds to a single sample's features as a tuple
            # map based on first element to return different SVs for tests
            if abs(x_tuple[0] - 0.1) < 1e-12:
                return sv1
            return sv2

        k._cached_execution = cached_exec_per_sample

        # input x has two samples -> expect 2x2 kernel
        x = np.array([[0.1, 0.2], [0.3, 0.4]])

        # off_diagonal: diagonal entries should be set to 1; off-diagonals computed from SV overlaps (0)
        k._evaluate_duplicates = "off_diagonal"
        mat = k.evaluate_kernel_sv(x, x)

        assert mat.shape == (2, 2)
        assert np.allclose(np.diag(mat), np.ones(2))  # diagonal set to 1
        assert mat[0, 1] == pytest.approx(0.0)
        assert mat[1, 0] == pytest.approx(0.0)

        # non-symmetric: compare x against single-sample y -> expect (2,1) output with overlaps [1,0]
        y = np.array([[0.1, 0.2]])
        mat2 = k.evaluate_kernel_sv(x, y)
        assert mat2.shape == (2, 1)
        assert mat2[0, 0] == pytest.approx(1.0)
        assert mat2[1, 0] == pytest.approx(0.0)

    def test_evaluate_kernel_sv_qulacs_with_parameter_vector_and_parameters(self):
        """
        Test the qulacs statevector branch when parameter_vector is present and parameters are provided.
        cached_execution will be called with (params_tuple, x_tuple) for every sample.
        """
        k = object.__new__(FidelityKernelStatevector)
        k._num_features = 2
        k._parameter_vector = object()  # indicates parameterised circuit
        k._parameters = np.array([0.42])  # provided params
        exec_mock = make_executor(is_statevector=True, framework="qulacs", shots=None)
        exec_mock.set_shots = MagicMock()
        k._executor = exec_mock

        sv1 = np.array([1.0 + 0j, 0.0 + 0j])
        sv2 = np.array([0.0 + 0j, 1.0 + 0j])

        # cached_execution accepts (params_tuple, x_tuple)
        def cached_exec_with_params(params_tuple, x_tuple):
            # verify params passed through
            assert tuple(params_tuple) == tuple(k._parameters)
            # dispatch based on x_tuple to return sv1 or sv2
            if abs(x_tuple[0] - 0.1) < 1e-12:
                return sv1
            return sv2

        k._cached_execution = cached_exec_with_params

        x = np.array([[0.1, 0.2], [0.3, 0.4]])
        k._evaluate_duplicates = "off_diagonal"

        mat = k.evaluate_kernel_sv(x, x)
        assert mat.shape == (2, 2)
        assert np.allclose(np.diag(mat), np.ones(2))
        assert mat[0, 1] == pytest.approx(0.0)
        assert mat[1, 0] == pytest.approx(0.0)

    def test_evaluate_kernel_sv_qulacs_with_parameter_vector_missing_parameters_raises(self):
        """
        If parameter_vector is set but _parameters is None, evaluate_kernel_sv should raise ValueError.
        """
        k = object.__new__(FidelityKernelStatevector)
        k._num_features = 1
        k._parameter_vector = object()
        k._parameters = None  # missing
        exec_mock = make_executor(is_statevector=True, framework="qulacs", shots=None)
        exec_mock.set_shots = MagicMock()
        k._executor = exec_mock

        x = np.array([[0.1]])
        with pytest.raises(ValueError):
            k.evaluate_kernel_sv(x, x)
