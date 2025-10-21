import numpy as np
import pytest

from squlearn.kernel.lowlevel_kernel import FidelityKernel


class DummyEncodingCircuit:
    def __init__(self):
        self._params = {"enc_param": 42}
        self.set_params_called_with = None
        self.num_qubits = 2
        self.num_parameters = len(self._params)

    def get_params(self):
        return dict(self._params)

    def set_params(self, **kwargs):
        # record calls so tests can assert
        self.set_params_called_with = dict(kwargs)
        self._params.update(kwargs)


class DummyExecutor:
    def __init__(self, framework="pennylane"):
        self.quantum_framework = framework
        self.backend_chosen = True
        self.is_statevector = True

    def get_shots(self):
        return 0

    def get_sampler(self):
        return None

    def select_backend(self, enc_circ, num_features):
        return enc_circ, None


class DummyQuantumKernel:

    def __init__(self, matrix=None):
        self._matrix = None if matrix is None else np.array(matrix, dtype=float)
        self.assigned_training_parameters = None
        self.evaluate_derivatives_called = None

    def assign_training_parameters(self, parameters):
        self.assigned_training_parameters = np.array(parameters, copy=True)

    def evaluate(self, x, y):
        # return a kernel matrix; if internal matrix is set use it, else return identity-like square
        if self._matrix is not None:
            return np.array(self._matrix, dtype=float)
        # default behavior: return small kernel depending on x,y sizes
        nx = len(x)
        ny = len(y)
        # create a matrix of 0.5 with ones on diagonal
        mat = 0.5 * np.ones((nx, ny))
        np.fill_diagonal(mat, 1.0 if nx == ny else 0.7)
        return mat

    def evaluate_derivatives(self, x, y, values):
        # record call and return a simple dict
        self.evaluate_derivatives_called = {"x": x, "y": y, "values": values}
        return {"dKdp": np.array([[0.1]])}


class TestFidelityKernel:

    def setup_method(self):
        # common fixtures
        self.enc = DummyEncodingCircuit()
        self.exec = DummyExecutor()
        self.kernel = FidelityKernel(
            encoding_circuit=self.enc, executor=self.exec, evaluate_duplicates="off_diagonal"
        )

    def test_get_params_deep_includes_encoding_params(self):
        params_shallow = self.kernel.get_params(deep=False)
        assert "evaluate_duplicates" in params_shallow
        assert params_shallow["evaluate_duplicates"] == "off_diagonal"

        params_deep = self.kernel.get_params(deep=True)
        # deep should include keys from encoding circuit.get_params()
        assert "enc_param" in params_deep
        assert params_deep["enc_param"] == 42

    def test_set_params_updates_encoding_and_reinitializes(self):
        # prepare new encoding param to pass via set_params
        new_value = 99
        # call set_params with a key that belongs to the encoding circuit
        self.kernel.set_params(enc_param=new_value)

        # encoding circuit should have been updated via set_params
        assert self.enc.set_params_called_with == {"enc_param": new_value}
        # the encoding internal param updated
        assert self.enc.get_params()["enc_param"] == new_value

        # setting an invalid parameter should raise ValueError
        with pytest.raises(ValueError):
            self.kernel.set_params(non_existing_param=1)

    def test_evaluate_requires_parameters_if_parameter_vector_present(self):
        # Patch kernel._initialize_kernel to simulate parameter_vector presence and a dummy quantum kernel
        def fake_init(num_features):
            self.kernel._parameter_vector = object()
            self.kernel._quantum_kernel = DummyQuantumKernel()
            self.kernel._is_initialized = True

        self.kernel._initialize_kernel = fake_init

        # no parameters assigned -> should raise ValueError
        with pytest.raises(ValueError):
            self.kernel.evaluate(x=np.array([[0.0]]), y=None)

        # now assign parameters and it should work and call assign_training_parameters
        self.kernel._parameters = np.array([0.1, 0.2])
        out = self.kernel.evaluate(x=np.array([[0.0], [1.0]]), y=None)
        # out should be a numpy array (kernel matrix) returned by DummyQuantumKernel.evaluate
        assert isinstance(out, np.ndarray)
        # assigned_training_parameters should have been set
        assert np.allclose(
            self.kernel._quantum_kernel.assigned_training_parameters, self.kernel._parameters
        )

    def test_evaluate_calls_get_msplit_when_mit_depol_noise_and_square(self):
        # prepare dummy quantum kernel that returns a known matrix
        dummy_matrix = np.array([[0.8, 0.2], [0.2, 0.9]])
        dqk = DummyQuantumKernel(matrix=dummy_matrix)

        # patch initialize to set the dummy quantum kernel
        def fake_init(num_features):
            self.kernel._parameter_vector = None
            self.kernel._quantum_kernel = dqk
            self.kernel._is_initialized = True

        self.kernel._initialize_kernel = fake_init

        # set mit_depol_noise and patch _get_msplit_kernel to observe it's invoked
        self.kernel._mit_depol_noise = "msplit"

        called = {"msplit": False}

        def fake_msplit(mat):
            called["msplit"] = True
            # return a trivially modified matrix to assert the pipeline returns this
            return mat + 0.1

        self.kernel._get_msplit_kernel = fake_msplit

        # evaluate with x != y -> should raise (per implementation) because mitigation works only for square matrices (and real backend)
        # The code raises only if not np.array_equal(x, y)
        with pytest.raises(ValueError):
            self.kernel.evaluate(x=np.array([[0.0], [1.0]]), y=np.array([[2.0]]))

        # evaluate with x == y -> should call _get_msplit_kernel and return modified matrix
        x = np.array([[0.0], [1.0]])
        out = self.kernel.evaluate(x=x, y=None)  # y defaults to x
        assert called["msplit"] is True
        assert np.allclose(out, dummy_matrix + 0.1)

    def test_evaluate_regularization_called_for_square_matrix(self):
        # use dummy quantum kernel returning a 2x2 matrix
        dqk = DummyQuantumKernel(matrix=np.array([[1.0, 0.0], [0.0, 1.0]]))

        def fake_init(num_features):
            self.kernel._quantum_kernel = dqk
            self.kernel._is_initialized = True
            self.kernel._parameter_vector = None

        self.kernel._initialize_kernel = fake_init

        # set regularization and patch _regularize_matrix
        self.kernel._regularization = "thresholding"
        called = {"regularize": False}

        def fake_regularize(mat):
            called["regularize"] = True
            return mat * 2.0

        self.kernel._regularize_matrix = fake_regularize

        out = self.kernel.evaluate(x=np.array([[0.0]] * 2), y=None)
        assert called["regularize"] is True
        assert np.allclose(out, np.array([[2.0, 0.0], [0.0, 2.0]]))

    def test_evaluate_derivatives_behaviour_for_expectation_flag(self):
        # prepare a dummy quantum kernel that implements evaluate_derivatives
        dqk = DummyQuantumKernel()

        def fake_init():
            self.kernel._is_initialized = True
            self.kernel._quantum_kernel = dqk
            self.kernel._parameter_vector = None

        self.kernel._initialize_kernel = fake_init
        self.kernel._initialize_kernel()

        # when _use_expectation is False -> should raise NotImplementedError
        self.kernel._use_expectation = False
        with pytest.raises(NotImplementedError):
            self.kernel.evaluate_derivatives(x=np.array([[0.0]]), y=None, values="dKdp")

        # when True -> should delegate to underlying quantum kernel
        self.kernel._use_expectation = True
        out = self.kernel.evaluate_derivatives(x=np.array([[0.0]]), y=None, values="dKdp")
        assert isinstance(out, dict)
        assert "dKdp" in out or dqk.evaluate_derivatives_called is not None
