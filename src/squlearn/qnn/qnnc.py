"""QNNClassifier Implemenation"""

from packaging import version
from typing import Callable, Union
import sys

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn import __version__

if version.parse(__version__) >= version.parse("1.6"):
    from sklearn.utils.validation import validate_data
else:

    def validate_data(self, *args, **kwargs):
        return self._validate_data(*args, **kwargs)


from tqdm import tqdm

from .base_qnn import BaseQNN
from .loss import VarianceLoss
from .loss.qnn_loss_base import QNNLossBase
from .util.training import train_mini_batch, train, ShotControlBase

from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ..optimizers.optimizer_base import OptimizerBase, SGDMixin
from ..util import Executor


class QNNClassifier(BaseQNN, ClassifierMixin):
    """Quantum Neural Network for Classification.

    This class implements a quantum neural network (QNN) for classification with a scikit-learn
    interface. A parameterized quantum circuit and a possibly parameterized operator are used
    as a ML model. They are trained according to a specified loss using the specified optimizer.
    Mini-batch training is possible.

    Args:
        encoding_circuit (EncodingCircuitBase): The parameterized quantum circuit (PQC) part of the QNN.
            For a list of encoding circuits, check this list of implemented :ref:`encoding_circuits`.
        operator (Union[ObservableBase, list[ObservableBase]]): The operator that
            is used in the expectation value of the QNN. Can be a list for multiple outputs. For a
            list of operators, check this list of implemented :ref:`operators`.
        executor (Executor): Executor instance.
        loss (QNNLossBase): The loss function to be optimized. Can also be combination of multiple
            loss functions.
        optimizer (OptimizerBase): The optimizer instance that is used to minimize the loss
            function.
        param_ini (np.ndarray, default=None): Initial values of the parameters of the PQC.
        param_op_ini (np.ndarray, default=None): Initial values of the parameters of the operator.
        batch_size (int, default=None): Number of data points in each batch in mini-batch training.
            Will only be used if optimizer is of type SGDMixin.
        epochs (int, default=None): Number of epochs of SGD to perform. Will only be used if
            optimizer is of type SGDMixin.
        shuffle (bool, default=None): If True, data points get shuffled before each epoch. Will
            only be used if optimizer is of type SGDMixin.
        opt_param_op (bool, default=True): If True, the operators parameters get optimized.
        variance (Union[float, Callable], default=None): The variance factor to be used. If it is
            None, the variance regularization will not be used. Else this determines the strength
            of the variance regularization.
        parameter_seed (Union[int, None], default=0): Seed for the random number generator for the
            parameter initialization, if `param_ini` or `param_op_ini` is ``None``.
        caching (bool, default=True): If True, the results of the QNN are cached.
        pretrained (bool, default=False): Set to true if the supplied parameters are already
                                          trained.
        callback (Union[Callable, str, None], default=None): A callback for the optimization loop.
            Can be either a Callable, "pbar" (which uses a :class:`tqdm.tqdm` process bar) or None.
            If None, the optimizers (default) callback will be used.
        primitive (Union[str,None], default=None): The primitive that is utilized in the qnn.
            Default primitive is the one specified in the executor initialization, if nothing is
            specified, the estimator will used. Possible values are ``"estimator"`` or
            ``"sampler"``.

    See Also
    --------
        squlearn.qnn.QNNRegressor : Quantum Neural Network for Regression.

    **Example**

    .. code-block::

        from squlearn import Executor
        from squlearn.encoding_circuit import ChebyshevRx
        from squlearn.observables import SummedPaulis
        from squlearn.qnn import QNNClassifier, SquaredLoss
        from squlearn.optimizers import SLSQP
        from sklearn.datasets import make_blobs
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        X, y = make_blobs(60, centers=2, random_state=0)
        X = MinMaxScaler((-0.9, 0.9)).fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        clf = QNNClassifier(
            ChebyshevRx(4, 2, 2),
            SummedPaulis(4),
            Executor(),
            SquaredLoss(),
            SLSQP(),
            np.random.rand(16),
            np.random.rand(5)
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    Methods:
    --------
    """

    def __init__(
        self,
        encoding_circuit: EncodingCircuitBase,
        operator: Union[ObservableBase, list[ObservableBase]],
        executor: Executor,
        loss: QNNLossBase,
        optimizer: OptimizerBase,
        param_ini: np.ndarray = None,
        param_op_ini: np.ndarray = None,
        batch_size: int = None,
        epochs: int = None,
        shuffle: bool = None,
        opt_param_op: bool = True,
        variance: Union[float, Callable] = None,
        shot_control: ShotControlBase = None,
        parameter_seed: Union[int, None] = 0,
        caching: bool = True,
        pretrained: bool = False,
        callback: Union[Callable, str, None] = "pbar",
        primitive: Union[str, None] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            encoding_circuit,
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
            shot_control,
            parameter_seed=parameter_seed,
            caching=caching,
            pretrained=pretrained,
            callback=callback,
            **kwargs,
        )
        self._label_binarizer = None
        self.classes_ = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the QNN.

        Args:
            X : The input data.

        Returns:
            np.ndarray : The predicted values.
        """
        X = validate_data(self, X, accept_sparse=["csr", "csc"], reset=False)

        if not self._is_fitted and not self.pretrained:
            raise RuntimeError("The model is not fitted.")

        if self.shot_control is not None:
            self.shot_control.reset_shots()

        pred = self._qnn.evaluate(X, self._param, self._param_op, "f")["f"]
        return self._label_binarizer.inverse_transform(pred)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities using the QNN.

        Args:
            X : The input data.

        Returns:
            np.ndarray : The probabilities
        """

        if self.shot_control is not None:
            self.shot_control.reset()

        pred = self._qnn.evaluate(X, self._param, self._param_op, "f")["f"]
        if pred.ndim == 1:
            return np.vstack([1 - pred, pred]).T

        return pred

    def partial_fit(self, X, y, weights: np.ndarray = None) -> None:
        """Fit a model to data.

        This method will update the models parameters to fit the provided data.
        It won't reinitialize the models parameters.

        Args:
            X: array-like or sparse matrix of shape (n_samples, n_features)
                Input data
            y: array-like of shape (n_samples,)
                Labels
            weights: Weights for each data point
        """
        X, y = self._validate_input(X, y, incremental=False, reset=False)

        if not self._is_fitted:
            self._label_binarizer = LabelBinarizer()
            y = self._label_binarizer.fit_transform(y)
            self.classes_ = self._label_binarizer.classes_
        else:
            y = self._label_binarizer.transform(y)

        if isinstance(self.operator, list) and len(self.operator) == 2 and y.shape[1] == 1:
            y = np.hstack([1 - y, y])

        if y.shape[1] == 1:
            y = self._label_binarizer.transform(y).ravel()

        loss = self.loss
        if self.variance is not None:
            loss = loss + VarianceLoss(alpha=self.variance)

        if isinstance(self.optimizer, SGDMixin) and self.batch_size:
            if self.opt_param_op:
                self._param, self._param_op = train_mini_batch(
                    self._qnn,
                    X,
                    y,
                    self._param,
                    self._param_op,
                    loss=loss,
                    optimizer=self.optimizer,
                    shot_control=self.shot_control,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    shuffle=self.shuffle,
                    weights=weights,
                    opt_param_op=True,
                )
            else:
                self._param = train_mini_batch(
                    self._qnn,
                    X,
                    y,
                    self._param,
                    self._param_op,
                    loss=loss,
                    optimizer=self.optimizer,
                    shot_control=self.shot_control,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    shuffle=self.shuffle,
                    weights=weights,
                    opt_param_op=False,
                )

        else:
            if self.opt_param_op:
                self._param, self._param_op = train(
                    self._qnn,
                    X,
                    y,
                    self._param,
                    self._param_op,
                    loss,
                    self.optimizer,
                    self.shot_control,
                    weights,
                    True,
                )
            else:
                self._param = train(
                    self._qnn,
                    X,
                    y,
                    self._param,
                    self._param_op,
                    loss,
                    self.optimizer,
                    self.shot_control,
                    weights,
                    False,
                )
        self._is_fitted = True

    def _fit(self, X, y, weights: np.ndarray = None) -> None:
        """Internal fit function.

        Args:
            X: array-like or sparse matrix of shape (n_samples, n_features)
                Input data
            y: array-like or sparse matrix of shape (n_samples,)
                Labels
            weights: Weights for each data point
        """
        if self.callback == "pbar":
            self._pbar = tqdm(total=self._total_iterations, desc="fit", file=sys.stdout)
        self.partial_fit(X, y, weights)
        if self.callback == "pbar":
            self._pbar.close()
