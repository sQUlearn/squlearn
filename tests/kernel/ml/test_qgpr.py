"""Tests for QGPR"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

from squlearn import Executor
from squlearn.encoding_circuit import YZ_CX_EncodingCircuit
from squlearn.kernel import QGPR
from squlearn.kernel.lowlevel_kernel import ProjectedQuantumKernel, FidelityKernel


class TestQGPR:
    """Test class for QGPR"""

    @pytest.fixture(scope="module")
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Test data module."""
        # pylint: disable=unbalanced-tuple-unpacking
        X, y = make_regression(n_samples=6, n_features=2, random_state=42)
        scl = MinMaxScaler((0.1, 0.9))
        X = scl.fit_transform(X, y)
        return X, y

    @pytest.fixture(scope="module")
    def qgpr_fidelity(self) -> QGPR:
        """QGPR module with FidelityKernel."""
        np.random.seed(42)
        executor = Executor()
        encoding_circuit = YZ_CX_EncodingCircuit(num_qubits=3, num_features=2, num_layers=2)
        kernel = FidelityKernel(encoding_circuit=encoding_circuit, executor=executor)
        return QGPR(quantum_kernel=kernel, sigma=1.0e-6)

    @pytest.fixture(scope="module")
    def qgpr_pqk(self) -> QGPR:
        """QGPR module with ProjectedQuantumKernel."""
        np.random.seed(42)
        executor = Executor()
        encoding_circuit = YZ_CX_EncodingCircuit(num_qubits=3, num_features=2, num_layers=2)
        kernel = ProjectedQuantumKernel(encoding_circuit=encoding_circuit, executor=executor)
        return QGPR(
            quantum_kernel=kernel, sigma=1.0e-6, normalize_y=False, full_regularization=True
        )

    def test_that_qgpr_params_are_present(self):
        """Asserts that all classical parameters are present in the QGPR."""
        qgpr_instance = QGPR(quantum_kernel=MagicMock())
        assert list(qgpr_instance.get_params(deep=False).keys()) == [
            "quantum_kernel",
            "sigma",
            "normalize_y",
            "full_regularization",
        ]

    @pytest.mark.parametrize("qgpr", ["qgpr_fidelity", "qgpr_pqk"])
    def test_predict(self, qgpr, request, data):
        """Tests concerning the predict function of the QGPR.

        Tests include
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qgpr_instance = request.getfixturevalue(qgpr)

        X, y = data
        qgpr_instance.fit(X, y)

        y_pred = qgpr_instance.predict(X)
        assert y_pred.shape == y.shape
        assert isinstance(y_pred, np.ndarray)

    @pytest.mark.parametrize("qgpr", ["qgpr_fidelity", "qgpr_pqk"])
    def test_list_input(self, qgpr, request, data):
        """Tests concerning the predict function of the QGPR with list input.

        Tests include
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qgpr_instance = request.getfixturevalue(qgpr)

        X, y = data
        qgpr_instance.fit(X.tolist(), y.tolist())

        y_pred = qgpr_instance.predict(X)
        assert y_pred.shape == y.shape
        assert isinstance(y_pred, np.ndarray)

    @pytest.mark.parametrize("qgpr", ["qgpr_fidelity", "qgpr_pqk"])
    def test_return_cov(self, qgpr, request, data):
        """Tests concerning the predict function of the QGPR.

        Tests include
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qgpr_instance = request.getfixturevalue(qgpr)

        X, y = data
        qgpr_instance.fit(X, y)

        y_pred, cov = qgpr_instance.predict(X, return_cov=True)
        assert y_pred.shape == y.shape
        assert isinstance(y_pred, np.ndarray)
        assert cov.shape[0] == cov.shape[1]

    @pytest.mark.parametrize("qgpr", ["qgpr_fidelity", "qgpr_pqk"])
    def test_return_std(self, qgpr, request, data):
        """Tests concerning the predict function of the QGPR.

        Tests include
            - whether the output is of the same shape as the reference
            - whether the type of the output is np.ndarray
        """
        qgpr_instance = request.getfixturevalue(qgpr)

        X, y = data
        qgpr_instance.fit(X, y)

        y_pred, std = qgpr_instance.predict(X, return_std=True)
        assert y_pred.shape == y.shape
        assert isinstance(y_pred, np.ndarray)
        assert std.shape[0] == X.shape[0]

    @pytest.mark.parametrize("qgpr", ["qgpr_fidelity", "qgpr_pqk"])
    def test_kernel_params_can_be_changed_after_initialization(self, qgpr, request, data):
        """Tests concerning the kernel parameter changes."""
        qgpr_instance = request.getfixturevalue(qgpr)

        qgpr_params = qgpr_instance.get_params()
        assert qgpr_params["num_qubits"] == 3
        assert qgpr_params["full_regularization"] == True

        qgpr_instance.set_params(num_qubits=4)
        qgpr_instance.set_params(full_regularization=False)

        qgpr_params_updated = qgpr_instance.get_params()
        assert qgpr_params_updated["num_qubits"] == 4
        assert qgpr_params_updated["full_regularization"] == False

        # check if fit is still possible
        X, y = data
        try:
            qgpr_instance.fit(X, y)
        except:
            assert False, f"fitting not possible after changes to quantum kernel parameters"

    @pytest.mark.parametrize("qgpr", ["qgpr_fidelity", "qgpr_pqk"])
    def test_encoding_circuit_params_can_be_changed_after_initialization(
        self, qgpr, request, data
    ):
        """Tests concerning the encoding circuit parameter changes."""
        qgpr_instance = request.getfixturevalue(qgpr)
        assert qgpr_instance.get_params()["num_layers"] == 2
        qgpr_instance.set_params(num_layers=4)
        assert qgpr_instance.get_params()["num_layers"] == 4

        # Check if fit is still possible
        X, y = data
        try:
            qgpr_instance.fit(X, y)
        except:
            assert False, f"fitting not possible after changes to encoding circuit paramaeters"

    def test_pqk_params_can_be_changes_after_initialization(self, qgpr_pqk, data):
        """Tests concerning changes if PQK parameters."""

        qgpr_params = qgpr_pqk.get_params()
        assert qgpr_params["gamma"] == 1.0
        assert qgpr_params["measurement"] == "XYZ"
        qgpr_pqk.set_params(gamma=0.5, measurement="Z")

        qgpr_params_updated = qgpr_pqk.get_params()
        assert qgpr_params_updated["gamma"] == 0.5
        assert qgpr_params_updated["measurement"] == "Z"

        # Check if fit is still possible
        X, y = data
        try:
            qgpr_pqk.fit(X, y)
        except:
            assert False, f"fitting not possible after changes of PQK paramaeters"

    @pytest.mark.parametrize("qgpr", ["qgpr_fidelity", "qgpr_pqk"])
    def test_classical_params_can_be_changed_after_initialization(self, qgpr, request):
        """Test concerning change of classical GPR parameter"""
        qgpr_instance = request.getfixturevalue(qgpr)

        qgpr_params = qgpr_instance.get_params()
        assert qgpr_params["sigma"] == 1.0e-6
        assert qgpr_params["normalize_y"] == False
        qgpr_instance.set_params(sigma=0.01)
        qgpr_instance.set_params(normalize_y=True)

        qgpr_params_updated = qgpr_instance.get_params()
        assert qgpr_params_updated["sigma"] == 0.01
        assert qgpr_params_updated["normalize_y"] == True

    @pytest.mark.parametrize("qgpr", ["qgpr_fidelity", "qgpr_pqk"])
    def test_that_regularization_is_called_when_not_none(self, qgpr, request, data):
        """Asserts that regularization is called."""
        qgpr_instance = request.getfixturevalue(qgpr)
        X, y = data

        qgpr_instance.set_params(regularization="tikhonov")

        qgpr_instance._quantum_kernel._regularize_matrix = MagicMock()
        qgpr_instance._quantum_kernel._regularize_matrix.side_effect = lambda x: x

        qgpr_instance.fit(X, y)
        qgpr_instance.predict(X)

        assert qgpr_instance._quantum_kernel._regularize_matrix.call_count == 3
