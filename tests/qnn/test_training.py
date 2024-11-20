"""Tests for training methods."""

import numpy as np
import pytest

from squlearn import Executor
from squlearn.observables import SummedPaulis, SinglePauli
from squlearn.encoding_circuit import ChebyshevPQC, HighDimEncodingCircuit
from squlearn.optimizers import SLSQP, Adam
from squlearn.qnn.loss import SquaredLoss
from squlearn.qnn.lowlevel_qnn import LowLevelQNN
from squlearn.qnn import QNNRegressor, QNNClassifier
from squlearn.qnn.training import train_mini_batch, ShotsFromRSTD

executor = Executor()

examples = [np.arange(0.1, 0.9, 0.01), np.log(np.arange(0.1, 0.9, 0.01))]


class TestSolvemini_batch:
    """Tests for mini-batch gradient descent."""

    pqc = ChebyshevPQC(4, 1, 3, False)
    cost_op = SummedPaulis(4)
    qnn = LowLevelQNN(pqc, cost_op, executor)
    ex_1 = [np.arange(0.1, 0.9, 0.01), np.log(np.arange(0.1, 0.9, 0.01))]

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

        pqc = ChebyshevPQC(2, 1, 3, False)
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

        reference = np.array([-0.23382872, -0.42261867, -0.17905442, 0.18627426, 0.09754079])
        assert np.allclose(test, reference, atol=1e-3)

    def test_qnn_training_two_outputs(self):
        """Test a optimization with variance reduction and shots from RSTD with two outputs."""

        pqc = ChebyshevPQC(2, 1, 3, False)
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
                [0.42006322, 0.39721762],
                [0.38582841, 0.3824161],
                [0.23203671, 0.20858907],
                [-0.0534579, -0.06126247],
                [-0.12367106, -0.14982011],
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

        assert_dict = {
            "QNNRegressor": np.array([0.11503425, 0.10989764, 0.11377155, 0.12618358, 0.14544058]),
            "QNNClassifier": np.array([0, 0, 0, 0, 0]),
        }
        pqc = ChebyshevPQC(2, 1, 1)
        ob = SinglePauli(2, 0, "Z")

        qnn, x_train, y_train = self._build_qnn_setup(pqc, ob, test_case)
        assert qnn.num_parameters_observable == 0
        qnn.fit(x_train, y_train)
        assert np.allclose(qnn.predict(x_train), assert_dict[test_case], atol=1e-6)

    @pytest.mark.parametrize("test_case", ["QNNRegressor", "QNNClassifier"])
    def test_zero_param(self, test_case):
        """Test for zero number of parameters in encoding circuit."""

        assert_dict = {
            "QNNRegressor": np.array([0.12, 0.12, 0.12, 0.12, 0.12]),
            "QNNClassifier": np.array([1, 0, 0, 0, 0]),
        }

        pqc = HighDimEncodingCircuit(2, 1, num_layers=1)
        ob = SummedPaulis(2)

        qnn, x_train, y_train = self._build_qnn_setup(pqc, ob, test_case)
        assert qnn.num_parameters == 0
        qnn.fit(x_train, y_train)
        assert np.allclose(qnn.predict(x_train), assert_dict[test_case], atol=1e-6)

    @pytest.mark.parametrize("test_case", ["QNNRegressor", "QNNClassifier"])
    def test_all_zero(self, test_case):
        """Test for zero number of parameters in both observable and encoding circuit."""

        assert_dict = {
            "QNNRegressor": np.array([0.19470917, 0.09933467, 0.0, -0.09933467, -0.19470917]),
            "QNNClassifier": np.array([0, 0, 0, 0, 0]),
        }

        pqc = HighDimEncodingCircuit(2, 1, num_layers=1)
        ob = SinglePauli(2, 0, "Z")

        qnn, x_train, y_train = self._build_qnn_setup(pqc, ob, test_case)
        assert qnn.num_parameters_observable == 0
        assert qnn.num_parameters == 0
        qnn.fit(x_train, y_train)
        assert np.allclose(qnn.predict(x_train), assert_dict[test_case], atol=1e-6)
