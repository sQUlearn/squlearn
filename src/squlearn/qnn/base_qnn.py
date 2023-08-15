"""QNN Base Implemenation"""
from __future__ import annotations

from abc import abstractmethod, ABC
import re
from typing import Callable, Union
from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator

from ..expectation_operator.expectation_operator_base import ExpectationOperatorBase
from ..feature_map.feature_map_base import FeatureMapBase
from ..optimizers.optimizer_base import OptimizerBase, SGDMixin
from ..util import Executor

from .loss import LossBase
from .qnn import QNN
from .training import shot_adjusting_options

MULTI_OP_KEY_PATTERN = re.compile(r"op(\d+)__(.*)")


class BaseQNN(BaseEstimator, ABC):
    """Base Class for Quantum Neural Networks.

    Args:
        feature_map : Parameterized quantum circuit in feature map format
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
        parameter_seed : Seed for the random number generator for the parameter initialization
    """

    def __init__(
        self,
        feature_map: FeatureMapBase,
        operator: Union[ExpectationOperatorBase, list[ExpectationOperatorBase]],
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
        shot_adjusting: shot_adjusting_options = None,
        parameter_seed: Union[int, None] = 0,
    ) -> None:
        super().__init__()
        self._feature_map = feature_map
        self._operator = operator
        self._loss = loss
        self._optimizer = optimizer
        self.variance = variance
        self.parameter_seed = parameter_seed

        if param_ini is None:
            self._param_ini = feature_map.generate_initial_parameters(seed=parameter_seed)
        else:
            self._param_ini = param_ini
        self._param = self._param_ini.copy()

        if param_op_ini is None:
            if isinstance(operator, list):
                self._param_op_ini = np.concatenate(
                    [
                        operator.generate_initial_parameters(seed=parameter_seed)
                        for operator in operator
                    ]
                )
            else:
                self._param_op_ini = operator.generate_initial_parameters(seed=parameter_seed)
        else:
            self._param_op_ini = param_op_ini
        self._param_op = self._param_op_ini.copy()

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

        self.shot_adjusting = shot_adjusting

        self._executor = executor
        self._qnn = QNN(self._feature_map, self._operator, executor)

        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> None:
        """Fit a new model to data.

        This method will reinitialize the models parameters and fit it to the provided data.

        Args:
            X: Input data
            y: Labels
            weights: Weights for each datapoint
        """
        self._param = self._param_ini.copy()
        self._param_op = self._param_op_ini.copy()
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
        params = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

        if deep:
            params.update(self._feature_map.get_params())
            if isinstance(self._operator, list):
                for i, oper in enumerate(self._operator):
                    oper_dict = oper.get_params()
                    for key, value in oper_dict.items():
                        if key != "num_qubits":
                            params[f"op{i}__{key}"] = value
            else:
                for key, value in self._operator.get_params().items():
                    if key != "num_qubits":
                        params[f"operator__{key}"] = value
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

        # Initialize parameter dictionaries
        if isinstance(self._operator, list):
            op_params = [{} for _ in range(len(self._operator))]
        else:
            op_params = {}
        feature_map_params = {}
        qnn_params = {}

        for key, value in params.items():
            # Check if parameter is valid
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )
            # Add num_qubits to all parameter dictionaries
            if key == "num_qubits":
                feature_map_params["num_qubits"] = value
                qnn_params["num_qubits"] = value
                if isinstance(self._operator, list):
                    for i in range(len(self._operator)):
                        op_params[i]["num_qubits"] = value
                else:
                    op_params["num_qubits"] = value
            # Add feature_map parameters to respective dictionary
            elif key in self._feature_map.get_params():
                feature_map_params[key] = value
                qnn_params[key] = value
            # Add operator parameters to respective dictionary
            # single op:
            elif key.startswith("operator__"):
                op_params[key[10:]] = value
                qnn_params[key[10:]] = value
            # multi op:
            else:
                match = MULTI_OP_KEY_PATTERN.match(key)
                if match:
                    op_params[int(match.group(1))][match.group(2)] = value
                    qnn_params[key] = value
                # Set parameter if not of any of the above
                else:
                    setattr(self, key, value)

        # Set all parameters for all objects
        if isinstance(self._operator, list):
            for i, operator in enumerate(self._operator):
                if op_params[i]:
                    operator.set_params(**op_params[i])
        elif op_params:
            self._operator.set_params(**op_params)

        if feature_map_params:
            self._feature_map.set_params(**feature_map_params)

        if qnn_params:
            self._qnn.set_params(**qnn_params)
            # If the number of parameters has changed, reinitialize the parameters
            if self._feature_map.num_parameters != len(self._param_ini):
                self._param_ini = self._feature_map.generate_initial_parameters(
                    seed=self.parameter_seed
                )
            if isinstance(self._operator, list):
                num_op_parameters = sum(operator.num_parameters for operator in self._operator)
                if num_op_parameters != len(self._param_op_ini):
                    self._param_op_ini = np.concatenate(
                        [
                            operator.generate_initial_parameters(seed=self.parameter_seed)
                            for operator in self._operator
                        ]
                    )
            elif self._operator.num_parameters != len(self._param_op_ini):
                self._param_op_ini = self._operator.generate_initial_parameters(
                    seed=self.parameter_seed
                )
            if isinstance(self._optimizer, SGDMixin):
                self._optimizer.reset()

        self._is_fitted = False

        return self

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> None:
        """Internal fit function."""
        raise NotImplementedError()
