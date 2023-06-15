"""QNNRegressor Implemenation"""
from typing import Callable, Union
from warnings import warn

import numpy as np
from sklearn.base import RegressorMixin

from .base_qnn import BaseQNN
from .loss import LossBase
from .training import solve_minibatch, regression

from ..expectation_operator.expectation_operator_base import ExpectationOperatorBase
from ..feature_map.feature_map_base import FeatureMapBase
from ..optimizers.optimizer_base import OptimizerBase, SGDMixin
from ..util import Executor


class QNNRegressor(BaseQNN, RegressorMixin):
    """Quantum Neural Network for Regression.

    This class implements a QNN for regression with a sklearn interface.

    Args:
        pqc : Parameterized quantum circuit in feature map format
        operator : Operator that are used in the expectation value of the QNN. Can be a list for
            multiple outputs.
        executor : Executor instance
        optimizer : Optimizer instance
        param_ini : Initialization values of the parameters of the PQC
        param_op_ini : Initialization values of the cost operator
        batch_size : Number of datapoints in each batch, for SGDMixin optimizers
        epochs : Number of epochs of SGD to perform, for SGDMixin optimizers
        shuffle : If True, datapoints get shuffled before each epoch (default: False),
            for SGDMixin optimizers
        opt_param_op : If True, operators parameters get optimized
        variance : Variance factor
    """

    def __init__(
        self,
        pqc: FeatureMapBase,
        operator: Union[ExpectationOperatorBase, list[ExpectationOperatorBase]],
        executor: Executor,
        loss: LossBase,
        optimizer: OptimizerBase,
        param_ini: np.ndarray = None,
        param_op_ini: np.ndarray = None,
        batch_size: int = None,
        epochs: int = None,
        shuffle: bool = None,
        opt_param_op: bool = True,
        variance: Union[float, Callable] = None,
    ) -> None:
        super().__init__(
            pqc,
            operator,
            executor,
            loss,
            optimizer,
            param_ini,
            param_op_ini,
            batch_size,
            epochs,
            shuffle,
            opt_param_op,
            variance,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the QNN.

        Args:
            X : The input data.

        Returns:
            np.ndarray : The predicted values.
        """
        if not self._is_fitted:
            warn("The model is not fitted.")
        return self._qnn.evaluate_f(X, self.param, self.param_op)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> None:
        """Fit a model to data.

        This method will update the models parameters to fit the provided data.
        It won't reinitialize the models parameters.

        Args:
            X: Input data
            y: Labels
            weights: Weights for each datapoint
        """
        if isinstance(self.optimizer, SGDMixin) and self.batch_size:
            if self.opt_param_op:
                self.param, self.param_op = solve_minibatch(
                    self._qnn,
                    X,
                    y,
                    self.param,
                    self.param_op,
                    loss=self.loss,
                    optimizer=self.optimizer,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    shuffle=self.shuffle,
                    weights=weights,
                    opt_param_op=True,
                    variance=self.variance,
                )
            else:
                self.param = solve_minibatch(
                    self._qnn,
                    X,
                    y,
                    self.param,
                    self.param_op,
                    loss=self.loss,
                    optimizer=self.optimizer,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    shuffle=self.shuffle,
                    weights=weights,
                    opt_param_op=False,
                    variance=self.variance,
                )

        else:
            if self.opt_param_op:
                self.param, self.param_op = regression(
                    self._qnn,
                    X,
                    y,
                    self.param,
                    self.param_op,
                    self.loss,
                    self.optimizer.minimize,
                    weights,
                    True,
                    self.variance,
                )
            else:
                self.param = regression(
                    self._qnn,
                    X,
                    y,
                    self.param,
                    self.param_op,
                    self.loss,
                    self.optimizer.minimize,
                    weights,
                    False,
                    self.variance,
                )
        self._is_fitted = True

    def _fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> None:
        """Internal fit function."""
        self.partial_fit(X, y, weights)
