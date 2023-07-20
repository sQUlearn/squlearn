"""Tests for training methods."""
import numpy as np

import pytest

from squlearn import Executor
from squlearn.expectation_operator import SummedPaulis
from squlearn.feature_map import ChebPQC
from squlearn.optimizers import Adam, SLSQP
from squlearn.qnn.loss import SquaredLoss
from squlearn.qnn.qnn import QNN
from squlearn.qnn.training import solve_minibatch

executor = Executor("statevector_simulator")

examples = [np.arange(0.1, 0.9, 0.01), np.log(np.arange(0.1, 0.9, 0.01))]


class TestSolveMinibatch:
    """Tests for minibatch gradient descent."""

    pqc = ChebPQC(4, 1, 3, False)
    cost_op = SummedPaulis(4)
    qnn = QNN(pqc, cost_op, executor)
    ex_1 = [np.arange(0.1, 0.9, 0.01), np.log(np.arange(0.1, 0.9, 0.01))]

    def test_wrong_optimizer(self):
        """Test for error caused by wrong optimizer type."""
        param_ini = np.random.rand(self.qnn.num_parameters) * 4
        param_op_ini = np.ones(self.qnn.num_qubits + 1)
        with pytest.raises(TypeError, match="is not supported for minibatch gradient descent."):
            solve_minibatch(
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
