"""Tests for QKRR"""

import io
import pytest
import numpy as np
from unittest.mock import MagicMock

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

from squlearn import Executor
from squlearn.encoding_circuit import HubregtsenEncodingCircuit
from squlearn.kernel import QKRR
from squlearn.kernel.lowlevel_kernel import ProjectedQuantumKernel, FidelityKernel


class TestQKRR:
    """Test class for QKRR"""

    @pytest.fixture(scope="module")
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Test data module."""
        # pylint: disable=unbalanced-tuple-unpacking
        X, y = make_regression(n_samples=6, n_features=2, random_state=42)
        scl = MinMaxScaler((0.1, 0.9))
        X = scl.fit_transform(X, y)
        return X, y

    @pytest.fixture
    def qkrr_fidelity(self) -> QKRR:
        """QKRR module with FidelityKernel."""
        np.random.seed(42)
        executor = Executor()
        encoding_circuit = HubregtsenEncodingCircuit(num_qubits=3, num_features=2, num_layers=2)
        kernel = FidelityKernel(
            encoding_circuit=encoding_circuit,
            executor=executor,
            regularization="thresholding",
            mit_depol_noise="msplit",
        )
        return QKRR(quantum_kernel=kernel, alpha=1.0e-6)

    @pytest.fixture
    def qkrr_pqk(self) -> QKRR:
        """QKRR module with ProjectedQuantumKernel."""
        np.random.seed(42)
        executor = Executor()
        encoding_circuit = HubregtsenEncodingCircuit(num_qubits=3, num_features=2, num_layers=2)
        kernel = ProjectedQuantumKernel(
            encoding_circuit=encoding_circuit, executor=executor, regularization="thresholding"
        )
        return QKRR(quantum_kernel=kernel, alpha=1.0e-6)

    @pytest.fixture(scope="module")
    def qkrr_fidelity_without_num_features(self) -> QKRR:
        """QKRR module with FidelityKernel."""
        np.random.seed(42)
        executor = Executor()
        encoding_circuit = HubregtsenEncodingCircuit(num_qubits=3, num_layers=2)
        kernel = FidelityKernel(
            encoding_circuit=encoding_circuit,
            executor=executor,
            regularization="thresholding",
            mit_depol_noise="msplit",
        )
        return QKRR(quantum_kernel=kernel, alpha=1.0e-6)

    @pytest.fixture(scope="module")
    def qkrr_pqk_without_num_features(self) -> QKRR:
        """QKRR module with ProjectedQuantumKernel."""
        np.random.seed(42)
        executor = Executor()
        encoding_circuit = HubregtsenEncodingCircuit(num_qubits=3, num_layers=2)
        kernel = ProjectedQuantumKernel(
            encoding_circuit=encoding_circuit, executor=executor, regularization="thresholding"
        )
        return QKRR(quantum_kernel=kernel, alpha=1.0e-6)

    def test_that_qkrr_params_are_present(self):
        """Asserts that all classical parameters are present in the QKRR."""
        qkrr_instance = QKRR(quantum_kernel=MagicMock())
        assert list(qkrr_instance.get_params(deep=False).keys()) == ["quantum_kernel", "alpha"]

    @pytest.mark.parametrize("qkrr", ["qkrr_fidelity", "qkrr_pqk"])
    def test_predict(self, qkrr, request, data):
        """Tests concerning the predict function of the QKRR.

        Tests include
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qkrr_instance = request.getfixturevalue(qkrr)

        X, y = data
        qkrr_instance.fit(X, y)

        y_pred = qkrr_instance.predict(X)
        assert y_pred.shape == y.shape
        assert isinstance(y_pred, np.ndarray)

    @pytest.mark.parametrize(
        "qkrr", ["qkrr_fidelity_without_num_features", "qkrr_pqk_without_num_features"]
    )
    def test_predict_without_num_features(self, qkrr, request, data):
        """Tests concerning the predict function of the QKRR.

        Tests include
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qkrr_instance = request.getfixturevalue(qkrr)

        X, y = data
        qkrr_instance.fit(X, y)

        y_pred = qkrr_instance.predict(X)
        assert y_pred.shape == y.shape
        assert isinstance(y_pred, np.ndarray)

    @pytest.mark.parametrize("qkrr", ["qkrr_fidelity", "qkrr_pqk"])
    def test_list_input(self, qkrr, request, data):
        """Tests concerning the predict function of the QKRR with list input.

        Tests include
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qkrr_instance = request.getfixturevalue(qkrr)

        X, y = data
        qkrr_instance.fit(X.tolist(), y.tolist())

        y_pred = qkrr_instance.predict(X)
        assert y_pred.shape == y.shape
        assert isinstance(y_pred, np.ndarray)

    @pytest.mark.parametrize("qkrr", ["qkrr_fidelity", "qkrr_pqk"])
    def test_kernel_params_can_be_changed_after_initialization(self, qkrr, request, data):
        """Tests concerning the kernel parameter changes."""
        qkrr_instance = request.getfixturevalue(qkrr)

        qkrr_params = qkrr_instance.get_params()
        assert qkrr_params["num_qubits"] == 3
        assert qkrr_params["regularization"] == "thresholding"
        qkrr_instance.set_params(num_qubits=4, regularization="tikhonov")

        qkrr_params_updated = qkrr_instance.get_params()
        assert qkrr_params_updated["num_qubits"] == 4
        assert qkrr_params_updated["regularization"] == "tikhonov"

        # check if fit is still possible
        X, y = data
        try:
            qkrr_instance.fit(X, y)
        except:
            assert False, f"fitting not possible after changes to quantum kernel parameters"

    @pytest.mark.parametrize("qkrr", ["qkrr_fidelity", "qkrr_pqk"])
    def test_encoding_circuit_params_can_be_changed_after_initialization(
        self, qkrr, request, data
    ):
        """Tests concerning the encoding circuit parameter changes."""
        qkrr_instance = request.getfixturevalue(qkrr)
        assert qkrr_instance.get_params()["num_layers"] == 2
        qkrr_instance.set_params(num_layers=4)

        assert qkrr_instance.get_params()["num_layers"] == 4

        # Check if fit is still possible
        X, y = data
        try:
            qkrr_instance.fit(X, y)
        except:
            assert False, f"fitting not possible after changes to encoding circuit paramaeters"

    def test_pqk_params_can_be_changes_after_initialization(self, qkrr_pqk, data):
        """Tests concerning changes if PQK parameters."""

        qkrr_params = qkrr_pqk.get_params()
        assert qkrr_params["gamma"] == 1.0
        assert qkrr_params["measurement"] == "XYZ"
        qkrr_pqk.set_params(gamma=0.5, measurement="Z")

        qkrr_params_updated = qkrr_pqk.get_params()
        assert qkrr_params_updated["gamma"] == 0.5
        assert qkrr_params_updated["measurement"] == "Z"

        # Check if fit is still possible
        X, y = data
        try:
            qkrr_pqk.fit(X, y)
        except:
            assert False, f"fitting not possible after changes of PQK paramaeters"

    @pytest.mark.parametrize("qkrr", ["qkrr_fidelity", "qkrr_pqk"])
    def test_classical_params_can_be_changed_after_initialization(self, qkrr, request):
        """Test concerning change of classical KRR parameter"""
        qkrr_instance = request.getfixturevalue(qkrr)

        qkrr_params = qkrr_instance.get_params()
        assert qkrr_params["alpha"] == 1.0e-6
        qkrr_instance.set_params(alpha=0.01)

        qkrr_params_updated = qkrr_instance.get_params()
        assert qkrr_params_updated["alpha"] == 0.01

    @pytest.mark.parametrize("qkrr", ["qkrr_fidelity", "qkrr_pqk"])
    def test_that_regularization_is_called_when_not_none(self, qkrr, request, data):
        """Asserts that regularization is called."""
        qkrr_instance = request.getfixturevalue(qkrr)
        X, y = data

        qkrr_instance.set_params(regularization="tikhonov")

        qkrr_instance._quantum_kernel._regularize_matrix = MagicMock()
        qkrr_instance._quantum_kernel._regularize_matrix.side_effect = lambda x: x

        qkrr_instance.fit(X, y)
        qkrr_instance.predict(X)

        assert qkrr_instance._quantum_kernel._regularize_matrix.call_count == 2

    @pytest.mark.parametrize("qkrr", ["qkrr_fidelity", "qkrr_pqk"])
    def test_serialization(self, qkrr, request, data):
        """Tests serialization of QKRR."""
        instance = request.getfixturevalue(qkrr)
        X, y = data
        instance.fit(X, y)

        buffer = io.BytesIO()
        instance.dump(buffer)

        predict_before = instance.predict(X)

        buffer.seek(0)
        instance_loaded = QKRR.load(buffer, Executor("qiskit"))
        predict_after = instance_loaded.predict(X)

        assert isinstance(instance_loaded, QKRR)
        assert np.allclose(predict_before, predict_after, atol=1e-6)

        instance_loaded.fit(X, y)
        predict_after2 = instance_loaded.predict(X)
        assert np.allclose(predict_before, predict_after2, atol=1e-6)
