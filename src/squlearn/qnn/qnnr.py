"""QNNRegressor Implemenation"""
from typing import Callable, Union
from warnings import warn

import numpy as np
from sklearn.base import RegressorMixin

from .base_qnn import BaseQNN
from .loss import LossBase, VarianceLoss
from .training import solve_minibatch, regression

from ..expectation_operator.expectation_operator_base import ExpectationOperatorBase
from ..feature_map.feature_map_base import FeatureMapBase
from ..optimizers.optimizer_base import OptimizerBase, SGDMixin
from ..util import Executor


class QNNRegressor(BaseQNN, RegressorMixin):
    """Quantum Neural Network for Regression.

    This class implements a quantum neural network (QNN) for regression with a sklearn interface.
    A parameterized quantum circuit and a possibly parameterized operator are used as a ML model.
    They are trained according to a specified loss using the specified optimizer. Minibatch
    training is possible.

    Args:
        pqc (FeatureMapBase): The parameterized quantum circuit (PQC) part of the QNN. For a list
            of feature maps, check the :ref:`Implemented feature maps in squlearn`.
        operator (Union[ExpectationOperatorBase, list[ExpectationOperatorBase]]): The operator that
            is used in the expectation value of the QNN. Can be a list for multiple outputs. For a
            list of operators, check the :ref:`Implemented operators for expectation values`
        executor (Executor): Executor instance.
        loss (LossBase): The loss function to be optimized. Can also be combination of multiple
            loss functions.
        optimizer (OptimizerBase): The optimizer instance that is used to minimize the loss
            function.
        param_ini (np.ndarray, default=None): Initial values of the parameters of the PQC.
        param_op_ini (np.ndarray, default=None): Initial values of the parameters of the operator.
        batch_size (int, default=None): Number of datapoints in each batch in minibatch training.
            Will only be used if optimizer is of type SGDMixin.
        epochs (int, default=None): Number of epochs of SGD to perform. Will only be used if
            optimizer is of type SGDMixin.
        shuffle (bool, default=None): If True, datapoints get shuffled before each epoch. Will only
            be used if optimizer is of type SGDMixin.
        opt_param_op (bool, default=True): If True, the operators parameters get optimized.
        variance (Union[float, Callable], default=None): The variance factor to be used. If it is
            None, the variance regularization will not be used. Else this determines the strength
            of the variance regularization.

    See Also
    --------
        squlearn.qnn.QNNClassifier : Quantum Neural Network for Classification.

    **Example**

    .. code-block::

        import numpy as np
        from squlearn import Executor
        from squlearn.feature_map import ChebRx
        from squlearn.expectation_operator import IsingHamiltonian
        from squlearn.qnn import QNNRegressor, SquaredLoss
        from squlearn.optimizers import SLSQP
        from sklearn.model_selection import train_test_split

        X, y = np.arange(0.1, 0.9, 0.01), np.log(np.arange(0.1, 0.9, 0.01))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        reg = QNNRegressor(
            ChebRx(4, 1, 2),
            IsingHamiltonian(4, I="S", Z="S", ZZ="S"),
            Executor("statevector_simulator"),
            SquaredLoss(),
            SLSQP(),
            np.random.rand(16),
            np.random.rand(5)
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test[:5])

    Methods:
    --------

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

        loss = self.loss
        if self.variance is not None:
            loss = loss + VarianceLoss(alpha=self.variance)

        if isinstance(self.optimizer, SGDMixin) and self.batch_size:
            if self.opt_param_op:
                self.param, self.param_op = solve_minibatch(
                    self._qnn,
                    X,
                    y,
                    self.param,
                    self.param_op,
                    loss=loss,
                    optimizer=self.optimizer,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    shuffle=self.shuffle,
                    weights=weights,
                    opt_param_op=True,
                )
            else:
                self.param = solve_minibatch(
                    self._qnn,
                    X,
                    y,
                    self.param,
                    self.param_op,
                    loss=loss,
                    optimizer=self.optimizer,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    shuffle=self.shuffle,
                    weights=weights,
                    opt_param_op=False,
                )

        else:
            if self.opt_param_op:
                self.param, self.param_op = regression(
                    self._qnn,
                    X,
                    y,
                    self.param,
                    self.param_op,
                    loss,
                    self.optimizer.minimize,
                    weights,
                    True,
                )
            else:
                self.param = regression(
                    self._qnn,
                    X,
                    y,
                    self.param,
                    self.param_op,
                    loss,
                    self.optimizer.minimize,
                    weights,
                    False,
                )
        self._is_fitted = True

    def _fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> None:
        """Internal fit function."""
        self.partial_fit(X, y, weights)
