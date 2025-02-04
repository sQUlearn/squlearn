"""QNNRegressor Implemenation"""

from packaging import version
from typing import Callable, Union
from warnings import warn
import sys

import numpy as np
from sklearn.base import RegressorMixin
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


class QNNRegressor(BaseQNN, RegressorMixin):
    """Quantum Neural Network for Regression.

    This class implements a quantum neural network (QNN) for regression with a scikit-learn
    interface. A parameterized quantum circuit and a possibly parameterized operator are used as
    a ML model. They are trained according to a specified loss using the specified optimizer.
    Mini-batch training is possible.

    Args:
        encoding_circuit (EncodingCircuitBase): The parameterized quantum circuit (PQC) part of the
            QNN. For a list of encoding circuits, check this list of implemented
            :ref:`encoding_circuits`.
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
        squlearn.qnn.QNNClassifier : Quantum Neural Network for Classification.

    **Example**

    .. code-block::

        import numpy as np
        from squlearn import Executor
        from squlearn.encoding_circuit import ChebyshevRx
        from squlearn.observables import IsingHamiltonian
        from squlearn.qnn import QNNRegressor, SquaredLoss
        from squlearn.optimizers import SLSQP
        from sklearn.model_selection import train_test_split

        X, y = np.arange(0.1, 0.9, 0.01), np.log(np.arange(0.1, 0.9, 0.01))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        reg = QNNRegressor(
            ChebyshevRx(4, 1, 2),
            IsingHamiltonian(4, I="S", Z="S", ZZ="S"),
            Executor(),
            SquaredLoss(),
            SLSQP(),
            np.random.rand(16),
            np.random.rand(5)
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test[:5])


    **Differential Evolution Solver Example**

    .. code-block::

        import numpy as np
        import sympy as sp
        import matplotlib.pyplot as plt
        from squlearn import Executor
        from squlearn.encoding_circuit import KyriienkoEncodingCircuit
        from squlearn.observables import SummedPaulis
        from squlearn.qnn import QNNRegressor, ODELoss
        from squlearn.qnn.util import get_lr_decay
        from squlearn.optimizers import Adam


        t, y, dydt, = sp.symbols("t y dydt")
        eq = 20 * sp.exp(-20 * t * 0.1) * sp.sin(20 * t) + 20 * 0.1 * y  + dydt
        initial_values = [1.0]

        loss_ODE = ODELoss(
            eq,
            symbols_involved_in_ODE=[t, y, dydt],
            initial_values=initial_values,
            boundary_handling="pinned",
        )

        circuit = KyriienkoEncodingCircuit(
            num_qubits=6,
            encoding_style="chebyshev_tower",
            variational_arrangement="HEA",
            num_features=1,
            num_encoding_layers=1,
            num_variational_layers=5,
        )
        observable = SummedPaulis(6, include_identity=False)

        param_observable = observable.generate_initial_parameters(seed=1)
        param_initial = circuit.generate_initial_parameters(seed=1)

        ode_regressor = QNNRegressor(
            circuit,
            observable,
            Executor("pennylane"),
            loss_ODE,
            Adam(options={"maxiter": 150, "tol": 0.00009, "lr": get_lr_decay(0.05, 0.02, 150)}),
            param_initial,
            param_observable,
            opt_param_op=False,
        )

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
            primitive=primitive,
            **kwargs,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the QNN.

        Args:
            X : The input data.

        Returns:
            np.ndarray : The predicted values.
        """
        X = validate_data(self, X, accept_sparse=["csr", "csc"], reset=False)

        if not self._is_fitted and not self.pretrained:
            warn("The model is not fitted.")

        if self.shot_control is not None:
            self.shot_control.reset_shots()

        return self._qnn.evaluate(X, self._param, self._param_op, "f")["f"]

    def partial_fit(self, X, y, weights: np.ndarray = None) -> None:
        """Fit a model to data.

        This method will update the models parameters to fit the provided data.
        It won't reinitialize the models parameters.

        Args:
            X: array-like or sparse matrix of shape (n_samples, n_features)
                Input data
            y: array-like or sparse matrix of shape (n_samples,)
                Labels
            weights: Weights for each data point
        """
        X, y = self._validate_input(X, y, incremental=False, reset=False)

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
