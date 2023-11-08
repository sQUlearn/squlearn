"""QNN Base Implemenation"""
from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Callable, Union
from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator


from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ..optimizers.optimizer_base import OptimizerBase, SGDMixin
from ..util import Executor

from .loss import LossBase
from .qnn import QNN
from .training import ShotControlBase


class BaseQNN(BaseEstimator, ABC):
    """Base Class for Quantum Neural Networks.

    Args:
        encoding_circuit : Parameterized quantum circuit in encoding circuit format
        operator : Operator that are used in the expectation value of the QNN. Can be a list for
            multiple outputs.
        executor : Executor instance
        optimizer : Optimizer instance
        param_ini : Initialization values of the parameters of the PQC
        param_op_ini : Initialization values of the cost operator
        batch_size : Number of data points in each batch, for SGDMixin optimizers
        epochs : Number of epochs of SGD to perform, for SGDMixin optimizers
        shuffle : If True, data points get shuffled before each epoch (default: False),
            for SGDMixin optimizers
        opt_param_op : If True, operators parameters get optimized
        variance : Variance factor
        parameter_seed : Seed for the random number generator for the parameter initialization
        caching : If True, the results of the QNN are cached.
        pretrained : Set to true if the supplied parameters are already trained.
        callback (Union[Callable, str, None], default=None): A callback for the optimization loop.
            Can be either a Callable, "pbar" (which uses a :class:`tqdm.tqdm` process bar) or None.
            If None, the optimizers (default) callback will be used.
    """

    def __init__(
        self,
        encoding_circuit: EncodingCircuitBase,
        operator: Union[ObservableBase, list[ObservableBase]],
        executor: Executor,
        loss: LossBase,
        optimizer: OptimizerBase,
        param_ini: Union[np.ndarray, None] = None,
        param_op_ini: Union[np.ndarray, None] = None,
        batch_size: int = None,
        epochs: int = None,
        shuffle: bool = None,
        opt_param_op: bool = True,
        variance: Union[float, Callable] = None,
        shot_control: ShotControlBase = None,
        parameter_seed: Union[int, None] = 0,
        caching: bool = True,
        pretrained: bool = False,
        callback: Union[Callable, str, None] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoding_circuit = encoding_circuit
        self.operator = operator
        self.loss = loss
        self.optimizer = optimizer
        self.variance = variance
        self.shot_control = shot_control
        self.parameter_seed = parameter_seed

        if param_ini is None:
            self.param_ini = encoding_circuit.generate_initial_parameters(seed=parameter_seed)
            if pretrained:
                raise ValueError("If pretrained is True, param_ini must be provided!")
        else:
            self.param_ini = param_ini
        self._param = self.param_ini.copy()

        if param_op_ini is None:
            if pretrained:
                raise ValueError("If pretrained is True, param_op_ini must be provided!")

            if isinstance(operator, list):
                self.param_op_ini = np.concatenate(
                    [
                        operator.generate_initial_parameters(seed=parameter_seed + i + 1)
                        for i, operator in enumerate(operator)
                    ]
                )
            else:
                self.param_op_ini = operator.generate_initial_parameters(seed=parameter_seed + 1)
        else:
            self.param_op_ini = param_op_ini
        self._param_op = self.param_op_ini.copy()

        if not isinstance(optimizer, SGDMixin) and any(
            param is not None for param in [batch_size, epochs, shuffle]
        ):
            warn(
                f"{optimizer.__class__.__name__} is not of type SGDMixin, thus batch_size, epochs"
                " and shuffle will be ignored."
            )
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle

        self.opt_param_op = opt_param_op

        self.caching = caching
        self.pretrained = pretrained

        self.executor = executor
        self._qnn = QNN(
            self.encoding_circuit, self.operator, executor, result_caching=self.caching
        )

        self.shot_control = shot_control
        if self.shot_control is not None:
            self.shot_control.set_executor(self.executor)

        self.callback = callback

        if self.callback:
            if callable(self.callback):
                self.optimizer.set_callback(self.callback)
            elif self.callback == "pbar":
                self._pbar = None
                if isinstance(self.optimizer, SGDMixin) and self.batch_size:
                    self._total_iterations = self.epochs
                else:
                    self._total_iterations = self.optimizer.options.get("maxiter", 100)

                def pbar_callback(*args):
                    self._pbar.update(1)

                self.optimizer.set_callback(pbar_callback)
            elif isinstance(self.callback, str):
                raise ValueError(f"Unknown callback string value {self.callback}")
            else:
                raise TypeError(f"Unknown callback type {type(self.callback)}")

        update_params = self.get_params().keys() & kwargs.keys()
        if update_params:
            self.set_params(**{key: kwargs[key] for key in update_params})

        self._is_fitted = self.pretrained

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_pbar"]
        return state

    def __setstate__(self, state) -> None:
        state.update({"_pbar": None})
        return super().__setstate__(state)

    @property
    def param(self) -> np.ndarray:
        """Parameters of the PQC."""
        return self._param

    @property
    def param_op(self) -> np.ndarray:
        """Parameters of the cost operator."""
        return self._param_op

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> None:
        """Fit a new model to data.

        This method will reinitialize the models parameters and fit it to the provided data.

        Args:
            X: Input data
            y: Labels
            weights: Weights for each data point
        """
        self._param = self.param_ini.copy()
        self._param_op = self.param_op_ini.copy()
        self._is_fitted = False
        self._fit(X, y, weights)

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns a dictionary of parameters for the current object.

        Parameters:
            deep: If True, includes the parameters from the base class.

        Returns:
            dict: A dictionary of parameters for the current object.
        """
        # Create a dictionary of all public parameters
        params = super().get_params(deep=False)

        if deep:
            params.update(self._qnn.get_params(deep=True))
        return params

    def set_params(self: BaseQNN, **params) -> BaseQNN:
        """
        Sets the hyper-parameters of the BaseQNN.

        Args:
            params: Hyper-parameters of the BaseQNN.

        Returns:
            updated BaseQNN
        """
        # Create dictionary of valid parameters
        valid_params = self.get_params().keys()
        for key in params.keys():
            # Check if parameter is valid
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

        # Set parameters
        self_params = params.keys() & self.get_params(deep=False).keys()
        for key in self_params:
            setattr(self, key, params[key])

        # Set parameters of the QNN
        qnn_params = params.keys() & self._qnn.get_params(deep=True).keys()
        if qnn_params:
            self._qnn.set_params(**{key: params[key] for key in qnn_params})

            # If the number of parameters has changed, reinitialize the parameters
            if self.encoding_circuit.num_parameters != len(self.param_ini):
                self.param_ini = self.encoding_circuit.generate_initial_parameters(
                    seed=self.parameter_seed
                )
            if isinstance(self.operator, list):
                num_op_parameters = sum(operator.num_parameters for operator in self.operator)
                if num_op_parameters != len(self.param_op_ini):
                    self.param_op_ini = np.concatenate(
                        [
                            operator.generate_initial_parameters(seed=self.parameter_seed)
                            for operator in self.operator
                        ]
                    )
            elif self.operator.num_parameters != len(self.param_op_ini):
                self.param_op_ini = self.operator.generate_initial_parameters(
                    seed=self.parameter_seed
                )
            if isinstance(self.optimizer, SGDMixin):
                self.optimizer.reset()

        self._is_fitted = False

        return self

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> None:
        """Internal fit function."""
        raise NotImplementedError()
