"""Tests for QSVC"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from squlearn import Executor
from squlearn.encoding_circuit import HubregtsenEncodingCircuit
from squlearn.kernel import QSVC
from squlearn.kernel.matrix import ProjectedQuantumKernel, FidelityKernel


class TestQSVC:
    """Test class for QSVC."""

    @pytest.fixture(scope="module")
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Test data module."""
        # pylint: disable=unbalanced-tuple-unpacking
        X, y = make_blobs(n_samples=6, n_features=2, centers=2, random_state=42)
        scl = MinMaxScaler((0.1, 0.9))
        X = scl.fit_transform(X, y)
        return X, y

    @pytest.fixture(scope="module")
    def qsvc_fidelity(self) -> QSVC:
        """QSVC module with FidelityKernel."""
        np.random.seed(42)
        executor = Executor()
        encoding_circuit = HubregtsenEncodingCircuit(num_qubits=3, num_features=2, num_layers=2)
        kernel = FidelityKernel(
            encoding_circuit,
            executor=executor,
            regularization="thresholding",
            mit_depol_noise="msplit",
        )
        return QSVC(kernel)

    @pytest.fixture(scope="module")
    def qsvc_pqk(self) -> QSVC:
        """QSVC module wit ProjectedQuantumKernel."""
        np.random.seed(42)
        executor = Executor()
        encoding_circuit = HubregtsenEncodingCircuit(num_qubits=3, num_features=2, num_layers=2)
        kernel = ProjectedQuantumKernel(
            encoding_circuit, executor=executor, regularization="thresholding"
        )
        return QSVC(kernel)

    def test_that_qsvc_params_are_present(self):
        """Asserts that all classical parameters are present in the QSVC."""
        qsvc_instance = QSVC(MagicMock())
        assert list(qsvc_instance.get_params(deep=False).keys()) == [
            "C",
            "break_ties",
            "cache_size",
            "class_weight",
            "decision_function_shape",
            "max_iter",
            "probability",
            "random_state",
            "shrinking",
            "tol",
            "verbose",
            "quantum_kernel",
        ]

    @pytest.mark.parametrize("qsvc", ["qsvc_fidelity", "qsvc_pqk"])
    def test_predict_unfitted(self, qsvc, request, data):
        """Tests concerning the unfitted QSVC.

        Tests include
            - whether a NotFittedError is raised
        """
        qsvc_instance = request.getfixturevalue(qsvc)
        X, _ = data
        with pytest.raises(NotFittedError):
            qsvc_instance.predict(X)

    @pytest.mark.parametrize("qsvc", ["qsvc_fidelity", "qsvc_pqk"])
    def test_predict(self, qsvc, request, data):
        """Tests concerning the predict function of the QSVC.

        Tests include
            - whether the prediction output is correct
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qsvc_instance = request.getfixturevalue(qsvc)

        X, y = data
        qsvc_instance.fit(X, y)

        y_pred = qsvc_instance.predict(X)
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y.shape
        assert np.allclose(y_pred, y)

    @pytest.mark.parametrize("qsvc", ["qsvc_fidelity", "qsvc_pqk"])
    def test_list_input(self, qsvc, request, data):
        """Tests concerning the predict function of the QSVC with list input.

        Tests include
            - whether the prediction output is correct
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qsvc_instance = request.getfixturevalue(qsvc)

        X, y = data
        qsvc_instance.fit(X.tolist(), y.tolist())

        y_pred = qsvc_instance.predict(X)
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y.shape
        assert np.allclose(y_pred, y)

    @pytest.mark.parametrize("qsvc", ["qsvc_fidelity", "qsvc_pqk"])
    def test_kernel_params_can_be_changed_after_initialization(self, qsvc, request, data):
        """Tests concerning the kernel parameter changes."""
        qsvc_instance = request.getfixturevalue(qsvc)

        qsvc_params = qsvc_instance.get_params()
        assert qsvc_params["num_qubits"] == 3
        assert qsvc_params["regularization"] == "thresholding"
        qsvc_instance.set_params(num_qubits=4, regularization="tikhonov")

        qsvc_params_updated = qsvc_instance.get_params()
        assert qsvc_params_updated["num_qubits"] == 4
        assert qsvc_params_updated["regularization"] == "tikhonov"

        # Check if fit is still possible
        X, y = data
        try:
            qsvc_instance.fit(X, y)
        except:
            assert False, f"fitting not possible after changes to quantum kernel parameters"

    @pytest.mark.parametrize("qsvc", ["qsvc_fidelity", "qsvc_pqk"])
    def test_encoding_circuit_params_can_be_changed_after_initialization(
        self, qsvc, request, data
    ):
        """Tests concerning the encoding circuit parameter changes."""
        qsvc_instance = request.getfixturevalue(qsvc)
        assert qsvc_instance.get_params()["num_layers"] == 2
        qsvc_instance.set_params(num_layers=4)
        assert qsvc_instance.get_params()["num_layers"] == 4

        # Check if fit is still possible
        X, y = data
        try:
            qsvc_instance.fit(X, y)
        except:
            assert False, f"fitting not possible after changes to encoding circuit parameters"

    def test_pqk_params_can_be_changed_after_initialization(self, qsvc_pqk, data):
        """Tests concerning the encoding circuit parameter changes."""
        qsvc_params = qsvc_pqk.get_params()
        assert qsvc_params["gamma"] == 1.0
        assert qsvc_params["measurement"] == "XYZ"
        qsvc_pqk.set_params(gamma=0.5, measurement="Z")

        qsvc_params_updated = qsvc_pqk.get_params()
        assert qsvc_params_updated["gamma"] == 0.5
        assert qsvc_params_updated["measurement"] == "Z"

        # Check if fit is still possible
        X, y = data
        try:
            qsvc_pqk.fit(X, y)
        except:
            assert False, f"fitting not possible after changes to encoding circuit parameters"

    @pytest.mark.parametrize("qsvc", ["qsvc_fidelity", "qsvc_pqk"])
    def test_classical_params_can_be_changed_after_initialization(self, qsvc, request):
        """Tests concerning the parameters of the classical SVC changes."""
        qsvc_instance = request.getfixturevalue(qsvc)
        assert qsvc_instance.get_params()["C"] == 1.0
        qsvc_instance.set_params(C=4)
        assert qsvc_instance.get_params()["C"] == 4

    @pytest.mark.parametrize("qsvc", ["qsvc_fidelity", "qsvc_pqk"])
    def test_that_regularization_is_called_when_not_none(self, qsvc, request, data):
        """Asserts that regularization is called."""
        qsvc_instance = request.getfixturevalue(qsvc)
        X, y = data

        qsvc_instance.set_params(regularization="tikhonov")

        qsvc_instance.quantum_kernel._regularize_matrix = MagicMock()
        qsvc_instance.quantum_kernel._regularize_matrix.side_effect = lambda x: x

        qsvc_instance.fit(X, y)
        qsvc_instance.predict(X)

        assert qsvc_instance.quantum_kernel._regularize_matrix.call_count == 2
