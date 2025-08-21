"""Tests for training methods."""

import numpy as np
import pytest
from packaging import version
from scipy import __version__ as scipy_version

from squlearn import Executor
from squlearn.observables import SummedPaulis, SinglePauli
from squlearn.encoding_circuit import ChebyshevPQC, HighDimEncodingCircuit
from squlearn.optimizers import SLSQP, Adam
from squlearn.qnn.loss import SquaredLoss
from squlearn.qnn.lowlevel_qnn import LowLevelQNN
from squlearn.qnn import QNNRegressor, QNNClassifier
from squlearn.qnn.util.training import train_mini_batch, ShotsFromRSTD

executor = Executor()

examples = [np.arange(0.1, 0.9, 0.01), np.log(np.arange(0.1, 0.9, 0.01))]


class TestSolvemini_batch:
    """Tests for mini-batch gradient descent."""

    pqc = ChebyshevPQC(num_qubits=4, closed=False)
    cost_op = SummedPaulis(4)
    ex_1 = [np.arange(0.1, 0.9, 0.01), np.log(np.arange(0.1, 0.9, 0.01))]
    qnn = LowLevelQNN(pqc, cost_op, executor, 1)

    def test_wrong_optimizer(self):
        """Test for error caused by wrong optimizer type."""
        param_ini = np.random.rand(self.qnn.num_parameters) * 4
        param_op_ini = np.ones(self.qnn.num_qubits + 1)
        with pytest.raises(TypeError, match="is not supported for mini-batch gradient descent."):
            train_mini_batch(
                self.qnn,
                self.ex_1[0],
                self.ex_1[1],
                param_ini,
                param_op_ini,
                loss=SquaredLoss(),
                optimizer=SLSQP(),
                batch_size=10,
                epochs=30,
            )


class TestShotsFromRSTD:
    """Tests for ShotsFromRSTD."""

    def test_qnn_training(self):
        """Test a optimization with variance reduction and shots from RSTD."""

        pqc = ChebyshevPQC(num_qubits=2, closed=False)
        ob = SummedPaulis(2)
        executor = Executor("qasm_simulator", seed=0)
        qnn = QNNRegressor(
            pqc,
            ob,
            executor,
            SquaredLoss(),
            Adam(options={"lr": 0.3, "maxiter": 3}),
            variance=0.005,
            shot_control=ShotsFromRSTD(),
            parameter_seed=0,
        )
        x_train = np.arange(-0.2, 0.3, 0.1).reshape(-1, 1)
        y_train = np.abs(x_train).ravel()
        qnn.fit(x_train, y_train)
        test = qnn.predict(x_train)
        reference = np.array([-0.09867318, -0.14080412, -0.1864203, -0.22726776, -0.24570661])
        assert np.allclose(test, reference, atol=1e-3)

    def test_qnn_training_two_outputs(self):
        """Test a optimization with variance reduction and shots from RSTD with two outputs."""

        pqc = ChebyshevPQC(num_qubits=2, closed=False)
        ob = [SummedPaulis(2), SummedPaulis(2)]
        executor = Executor("qasm_simulator", seed=0)
        qnn = QNNRegressor(
            pqc,
            ob,
            executor,
            SquaredLoss(),
            Adam(options={"lr": 0.3, "maxiter": 3}),
            variance=0.005,
            shot_control=ShotsFromRSTD(),
            parameter_seed=0,
        )
        x_train = np.arange(-0.2, 0.3, 0.1).reshape(-1, 1)
        y_train = np.hstack([np.abs(x_train), np.square(x_train)])
        qnn.fit(x_train, y_train)
        test = qnn.predict(x_train)
        reference = np.array(
            [
                [-0.10152712, -0.15280509],
                [-0.14254195, -0.19631623],
                [-0.19140704, -0.24552197],
                [-0.23021383, -0.28467866],
                [-0.24711364, -0.30817718],
            ]
        )

        assert np.allclose(test, reference, atol=1e-3)


class TestZeroParam:
    """Tests for zero number of parameters in both observable and encoding circuit."""

    def _build_qnn_setup(self, pqc, ob, test_case: str):
        """Helper function to build the qnn setup.

        Args:
            pqc (PQC): encoding circuit
            ob (Observable): observable
            test_case (str): test case type

        """
        executor = Executor("statevector_simulator")

        if test_case == "QNNRegressor":
            qnn = QNNRegressor(
                pqc,
                ob,
                executor,
                SquaredLoss(),
                SLSQP({"maxiter": 10}),
                variance=0.005,
                parameter_seed=0,
            )
            x_train = np.arange(-0.2, 0.3, 0.1).reshape(-1, 1)
            y_train = np.abs(x_train).ravel()
        else:
            qnn = QNNClassifier(
                pqc,
                ob,
                executor,
                SquaredLoss(),
                SLSQP({"maxiter": 10}),
                variance=0.005,
                parameter_seed=0,
            )
            x_train = np.arange(-0.2, 0.3, 0.1).reshape(-1, 1)
            y_train = np.array([0, 1, 1, 0, 0]).ravel()

        return qnn, x_train, y_train

    @pytest.mark.parametrize("test_case", ["QNNRegressor", "QNNClassifier"])
    def test_zero_param_ob(self, test_case):
        """Test for zero number of parameters in observable."""

        # scipy changed their slsqp implementation in 1.16.0 and we don't want to limit the user to
        # a specific scipy
        if version.parse(scipy_version) < version.parse("1.16.0"):
            regressor_result = np.array(
                [0.11503425, 0.10989764, 0.11377155, 0.12618358, 0.14544058]
            )
        else:
            regressor_result = np.array(
                [0.11080395, 0.10440662, 0.10877287, 0.12350483, 0.14668292]
            )

        assert_dict = {
            "QNNRegressor": regressor_result,
            "QNNClassifier": np.array([0, 0, 0, 0, 0]),
        }
        pqc = ChebyshevPQC(num_qubits=2)
        ob = SinglePauli(2, 0, "Z")

        qnn, x_train, y_train = self._build_qnn_setup(pqc, ob, test_case)
        qnn.fit(x_train, y_train)
        assert qnn.num_parameters_observable == 0
        assert np.allclose(qnn.predict(x_train), assert_dict[test_case], atol=1e-6)

    @pytest.mark.parametrize("test_case", ["QNNRegressor", "QNNClassifier"])
    def test_zero_param(self, test_case):
        """Test for zero number of parameters in encoding circuit."""

        assert_dict = {
            "QNNRegressor": np.array([0.12, 0.12, 0.12, 0.12, 0.12]),
            "QNNClassifier": np.array([0, 0, 0, 0, 0]),
        }

        pqc = HighDimEncodingCircuit(num_qubits=2, num_layers=1)
        ob = SummedPaulis(2)

        qnn, x_train, y_train = self._build_qnn_setup(pqc, ob, test_case)
        qnn.fit(x_train, y_train)
        assert qnn.num_parameters == 0
        assert np.allclose(qnn.predict(x_train), assert_dict[test_case], atol=1e-6)

    @pytest.mark.parametrize("test_case", ["QNNRegressor", "QNNClassifier"])
    def test_all_zero(self, test_case):
        """Test for zero number of parameters in both observable and encoding circuit."""

        assert_dict = {
            "QNNRegressor": np.array([0.19470917, 0.09933467, 0.0, -0.09933467, -0.19470917]),
            "QNNClassifier": np.array([1, 1, 0, 0, 0]),
        }

        pqc = HighDimEncodingCircuit(num_qubits=2, num_layers=1)
        ob = SinglePauli(2, 0, "Z")

        qnn, x_train, y_train = self._build_qnn_setup(pqc, ob, test_case)
        qnn.fit(x_train, y_train)
        assert qnn.num_parameters_observable == 0
        assert qnn.num_parameters == 0
        assert np.allclose(qnn.predict(x_train), assert_dict[test_case], atol=1e-6)
