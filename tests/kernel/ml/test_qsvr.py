"""Tests for QSVR"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from squlearn import Executor
from squlearn.encoding_circuit import MultiControlEncodingCircuit
from squlearn.kernel import QSVR
from squlearn.kernel.lowlevel_kernel import ProjectedQuantumKernel, FidelityKernel


class TestQSVR:
    """Test class for QSVR."""

    @pytest.fixture(scope="module")
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Test data module."""
        # pylint: disable=unbalanced-tuple-unpacking
        X, y = make_regression(n_samples=6, n_features=2, random_state=42)
        scl = MinMaxScaler((0.1, 0.9))
        X = scl.fit_transform(X, y)
        return X, y

    @pytest.fixture(scope="module")
    def qsvr_fidelity(self) -> QSVR:
        """QSVR module with FidelityKernel."""
        np.random.seed(42)
        executor = Executor()
        encoding_circuit = MultiControlEncodingCircuit(num_qubits=3, num_features=2, num_layers=2)
        kernel = FidelityKernel(
            encoding_circuit,
            executor=executor,
            regularization="thresholding",
            mit_depol_noise="msplit",
        )
        return QSVR(kernel, C=1, epsilon=0.1)

    @pytest.fixture(scope="module")
    def qsvr_pqk(self) -> QSVR:
        """QSVR module wit ProjectedQuantumKernel."""
        np.random.seed(42)
        executor = Executor()
        encoding_circuit = MultiControlEncodingCircuit(num_qubits=3, num_features=2, num_layers=2)
        kernel = ProjectedQuantumKernel(
            encoding_circuit, executor=executor, regularization="thresholding"
        )
        return QSVR(kernel, C=1, epsilon=0.1)

    def test_that_qsvr_params_are_present(self):
        """Asserts that all classical parameters are present in the QSVR."""
        qsvr_instance = QSVR(MagicMock())
        assert list(qsvr_instance.get_params(deep=False).keys()) == [
            "C",
            "cache_size",
            "epsilon",
            "max_iter",
            "shrinking",
            "tol",
            "verbose",
            "quantum_kernel",
        ]

    @pytest.mark.parametrize("qsvr", ["qsvr_fidelity", "qsvr_pqk"])
    def test_predict_unfitted(self, qsvr, request, data):
        """Tests concerning the unfitted QSVR.

        Tests include
            - whether a NotFittedError is raised
        """
        qsvr_instance = request.getfixturevalue(qsvr)
        X, _ = data
        with pytest.raises(NotFittedError):
            qsvr_instance.predict(X)

    @pytest.mark.parametrize("qsvr", ["qsvr_fidelity", "qsvr_pqk"])
    def test_predict(self, qsvr, request, data):
        """Tests concerning the predict function of the QSVR.

        Tests include
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qsvr_instance = request.getfixturevalue(qsvr)

        X, y = data
        qsvr_instance.fit(X, y)

        y_pred = qsvr_instance.predict(X)
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y.shape

    @pytest.mark.parametrize("qsvr", ["qsvr_fidelity", "qsvr_pqk"])
    def test_list_input(self, qsvr, request, data):
        """Tests concerning the predict function of the QSVR with list input.

        Tests include
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qsvr_instance = request.getfixturevalue(qsvr)

        X, y = data
        qsvr_instance.fit(X.tolist(), y.tolist())

        y_pred = qsvr_instance.predict(X)
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y.shape

    @pytest.mark.parametrize("qsvr", ["qsvr_fidelity", "qsvr_pqk"])
    def test_kernel_params_can_be_changed_after_initialization(self, qsvr, request, data):
        """Tests concerning the kernel parameter changes."""
        qsvr_instance = request.getfixturevalue(qsvr)

        qsvr_params = qsvr_instance.get_params()
        assert qsvr_params["num_qubits"] == 3
        assert qsvr_params["regularization"] == "thresholding"
        qsvr_instance.set_params(num_qubits=4, regularization="tikhonov")

        qsvr_params_updated = qsvr_instance.get_params()
        assert qsvr_params_updated["num_qubits"] == 4
        assert qsvr_params_updated["regularization"] == "tikhonov"

        # Check if fit is still possible
        X, y = data
        try:
            qsvr_instance.fit(X, y)
        except:
            assert False, f"fitting not possible after changes to quantum kernel parameters"

    @pytest.mark.parametrize("qsvr", ["qsvr_fidelity", "qsvr_pqk"])
    def test_encoding_circuit_params_can_be_changed_after_initialization(
        self, qsvr, request, data
    ):
        """Tests concerning the encoding circuit parameter changes."""
        qsvr_instance = request.getfixturevalue(qsvr)
        assert qsvr_instance.get_params()["num_layers"] == 2
        qsvr_instance.set_params(num_layers=4)
        assert qsvr_instance.get_params()["num_layers"] == 4

        # Check if fit is still possible
        X, y = data
        try:
            qsvr_instance.fit(X, y)
        except:
            assert False, f"fitting not possible after changes to encoding circuit parameters"

    def test_pqk_params_can_be_changed_after_initialization(self, qsvr_pqk, data):
        """Tests concerning the PQK parameter changes."""

        qsvr_params = qsvr_pqk.get_params()
        assert qsvr_params["gamma"] == 1.0
        assert qsvr_params["measurement"] == "XYZ"
        qsvr_pqk.set_params(gamma=0.5, measurement="Z")

        qsvr_params_updated = qsvr_pqk.get_params()
        assert qsvr_params_updated["gamma"] == 0.5
        assert qsvr_params_updated["measurement"] == "Z"

        # Check if fit is still possible
        X, y = data
        try:
            qsvr_pqk.fit(X, y)
        except:
            assert False, f"fitting not possible after changes to encoding circuit parameters"

    @pytest.mark.parametrize("qsvr", ["qsvr_fidelity", "qsvr_pqk"])
    def test_classical_params_can_be_changed_after_initialization(self, qsvr, request):
        """Tests concerning the parameters of the classical SVR changes."""
        qsvr_instance = request.getfixturevalue(qsvr)

        qsvr_params = qsvr_instance.get_params()
        assert qsvr_params["C"] == 1.0
        assert qsvr_params["epsilon"] == 0.1
        qsvr_instance.set_params(C=4, epsilon=0.5)

        qsvr_params_updated = qsvr_instance.get_params()
        assert qsvr_params_updated["C"] == 4
        assert qsvr_params_updated["epsilon"] == 0.5

    @pytest.mark.parametrize("qsvr", ["qsvr_fidelity", "qsvr_pqk"])
    def test_that_regularization_is_called_when_not_none(self, qsvr, request, data):
        """Asserts that regularization is called."""
        qsvr_instance = request.getfixturevalue(qsvr)
        X, y = data

        qsvr_instance.set_params(regularization="tikhonov")

        qsvr_instance.quantum_kernel._regularize_matrix = MagicMock()
        qsvr_instance.quantum_kernel._regularize_matrix.side_effect = lambda x: x

        qsvr_instance.fit(X, y)
        qsvr_instance.predict(X)

        assert qsvr_instance.quantum_kernel._regularize_matrix.call_count == 2
