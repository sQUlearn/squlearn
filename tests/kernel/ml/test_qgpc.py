"""Tests for QGPC"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from squlearn import Executor
from squlearn.encoding_circuit import HubregtsenEncodingCircuit
from squlearn.kernel import QGPC
from squlearn.kernel.lowlevel_kernel import ProjectedQuantumKernel, FidelityKernel


class TestQGPC:
    """Test class for QGPC."""

    @pytest.fixture(scope="module")
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Test data module."""
        # pylint: disable=unbalanced-tuple-unpacking
        X, y = make_blobs(n_samples=6, n_features=2, centers=2, random_state=42)
        scl = MinMaxScaler((0.1, 0.9))
        X = scl.fit_transform(X, y)
        return X, y

    @pytest.fixture(scope="module")
    def qgpc_fidelity(self) -> QGPC:
        """QGPC module with FidelityKernel."""
        np.random.seed(42)
        executor = Executor()
        encoding_circuit = HubregtsenEncodingCircuit(num_qubits=3, num_features=2, num_layers=2)
        kernel = FidelityKernel(
            encoding_circuit,
            executor=executor,
            regularization="thresholding",
            mit_depol_noise="msplit",
        )
        return QGPC(quantum_kernel=kernel)

    @pytest.fixture(scope="module")
    def qgpc_pqk(self) -> QGPC:
        """QGPC module wit ProjectedQuantumKernel."""
        np.random.seed(42)
        executor = Executor()
        encoding_circuit = HubregtsenEncodingCircuit(num_qubits=3, num_features=2, num_layers=2)
        kernel = ProjectedQuantumKernel(
            encoding_circuit, executor=executor, regularization="thresholding"
        )
        return QGPC(quantum_kernel=kernel)

    def test_that_qgpc_params_are_present(self):
        """Asserts that all classical parameters are present in the QGPC."""
        qgpc_instance = QGPC(quantum_kernel=MagicMock())
        assert list(qgpc_instance.get_params(deep=False).keys()) == [
            "copy_X_train",
            "max_iter_predict",
            "multi_class",
            "n_jobs",
            "n_restarts_optimizer",
            "optimizer",
            "random_state",
            "quantum_kernel",
        ]

    @pytest.mark.parametrize("qgpc", ["qgpc_fidelity", "qgpc_pqk"])
    def test_predict_unfitted(self, qgpc, request, data):
        """Tests concerning the unfitted QGPC.

        Tests include
            - whether a NotFittedError is raised
        """
        qgpc_instance = request.getfixturevalue(qgpc)
        X, _ = data
        with pytest.raises(NotFittedError):
            qgpc_instance.predict(X)

    @pytest.mark.parametrize("qgpc", ["qgpc_fidelity", "qgpc_pqk"])
    def test_predict(self, qgpc, request, data):
        """Tests concerning the predict function of the QGPC.

        Tests include
            - whether the prediction output is correct
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qgpc_instance = request.getfixturevalue(qgpc)

        X, y = data
        qgpc_instance.fit(X, y)

        y_pred = qgpc_instance.predict(X)
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y.shape
        assert np.allclose(y_pred, y)

    @pytest.mark.parametrize("qgpc", ["qgpc_fidelity", "qgpc_pqk"])
    def test_list_input(self, qgpc, request, data):
        """Tests concerning the predict function of the QGPC with list input.

        Tests include
            - whether the prediction output is correct
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qgpc_instance = request.getfixturevalue(qgpc)

        X, y = data
        qgpc_instance.fit(X.tolist(), y.tolist())

        y_pred = qgpc_instance.predict(X)
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y.shape
        assert np.allclose(y_pred, y)

    @pytest.mark.parametrize("qgpc", ["qgpc_fidelity", "qgpc_pqk"])
    def test_predict_probability(self, qgpc, request, data):
        """Tests concerning the predict function of the QGPC.

        Tests include
            - whether the prediction output is correct
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qgpc_instance = request.getfixturevalue(qgpc)

        X, y = data
        qgpc_instance.fit(X, y)

        y_prob = qgpc_instance.predict(X)
        assert isinstance(y_prob, np.ndarray)
        assert y_prob.shape == y.shape

    @pytest.mark.parametrize("qgpc", ["qgpc_fidelity", "qgpc_pqk"])
    def test_kernel_params_can_be_changed_after_initialization(self, qgpc, request, data):
        """Tests concerning the kernel parameter changes."""
        qgpc_instance = request.getfixturevalue(qgpc)

        qgpc_params = qgpc_instance.get_params()
        assert qgpc_params["num_qubits"] == 3
        assert qgpc_params["regularization"] == "thresholding"
        qgpc_instance.set_params(num_qubits=4, regularization="tikhonov")

        qgpc_params_updated = qgpc_instance.get_params()
        assert qgpc_params_updated["num_qubits"] == 4
        assert qgpc_params_updated["regularization"] == "tikhonov"

        # Check if fit is still possible
        X, y = data
        try:
            qgpc_instance.fit(X, y)
        except:
            assert False, "fitting not possible after changes to quantum kernel parameters"

    @pytest.mark.parametrize("qgpc", ["qgpc_fidelity", "qgpc_pqk"])
    def test_encoding_circuit_params_can_be_changed_after_initialization(
        self, qgpc, request, data
    ):
        """Tests concerning the encoding circuit parameter changes."""
        qgpc_instance = request.getfixturevalue(qgpc)
        assert qgpc_instance.get_params()["num_layers"] == 2
        qgpc_instance.set_params(num_layers=4)
        assert qgpc_instance.get_params()["num_layers"] == 4

        # Check if fit is still possible
        X, y = data
        try:
            qgpc_instance.fit(X, y)
        except:
            assert False, "fitting not possible after changes to encoding circuit parameters"

    def test_pqk_params_can_be_changed_after_initialization(self, qgpc_pqk, data):
        """Tests concerning the encoding circuit parameter changes."""
        qgpc_params = qgpc_pqk.get_params()
        assert qgpc_params["gamma"] == 1.0
        assert qgpc_params["measurement"] == "XYZ"
        qgpc_pqk.set_params(gamma=0.5, measurement="Z")

        qgpc_params_updated = qgpc_pqk.get_params()
        assert qgpc_params_updated["gamma"] == 0.5
        assert qgpc_params_updated["measurement"] == "Z"

        # Check if fit is still possible
        X, y = data
        try:
            qgpc_pqk.fit(X, y)
        except:
            assert False, "fitting not possible after changes to encoding circuit parameters"

    @pytest.mark.parametrize("qgpc", ["qgpc_fidelity", "qgpc_pqk"])
    def test_classical_params_can_be_changed_after_initialization(self, qgpc, request):
        """Tests concerning the parameters of the classical GPC changes."""
        qgpc_instance = request.getfixturevalue(qgpc)
        assert qgpc_instance.get_params()["max_iter_predict"] == 100
        qgpc_instance.set_params(max_iter_predict=50)
        assert qgpc_instance.get_params()["max_iter_predict"] == 50
