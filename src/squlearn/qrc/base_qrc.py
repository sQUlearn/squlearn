import numpy as np
from typing import Union, List
from abc import abstractmethod, ABC

from sklearn.base import BaseEstimator
from sklearn.utils import column_or_1d
from qiskit.quantum_info.random import random_pauli_list

from ..observables.observable_base import ObservableBase
from ..observables import CustomObservable, SinglePauli
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ..util import Executor
from ..qnn.lowlevel_qnn import LowLevelQNN


class BaseQRC(BaseEstimator, ABC):
    """Base class for Quantum Reservoir Computing (QRC) models.

    Args:
        encoding_circuit (EncodingCircuitBase): The encoding circuit to use for encoding the data
            into the reservoir.
        executor (Executor): Executor instance
        ml_model (str): The classical machine learning model to use (default: linear), possible
            values are ``"mlp"``, ``"linear"``, and ``"kernel"``. Implementation depends on the
            child.
        ml_model_options (dict): The options for the machine learning model. Default options of the
            sklearn model are used if None.
        operators (Union[ObservableBase, list[ObservableBase], str]): Strategy for generating the
            operators used to measure the quantum reservoir. Possible values are:

                * ``"random_paulis"`` generates random Pauli operators (default).
                * ``"single_paulis"`` generates single qubit Pauli operators.

            Alternatively, a list of ObservableBase objects can be provided.
        num_operators (int): The number of random Pauli operators to generate for
            ``"operators = random_paulis"`` (default: 100).
        operator_seed (int): The seed for the random operator generation for
            ``"operators = random_paulis"`` (default: 0).
        param_ini (Union[np.ndarray, None]): The parameters for the encoding circuit.
        param_op_ini (Union[np.ndarray, None]): The initial parameters for the operators.
        parameter_seed (Union[int, None]): The seed for the initial parameter generation if no
            parameters are given.
        caching (bool): Whether to cache the results of the evaluated expectation values.
    """

    def __init__(
        self,
        encoding_circuit: EncodingCircuitBase,
        executor: Executor,
        ml_model: str = "linear",
        ml_model_options: Union[dict, None] = None,
        operators: Union[ObservableBase, list[ObservableBase], str] = "random_paulis",
        num_operators: int = 100,
        operator_seed: int = 0,
        param_ini: Union[np.ndarray, None] = None,
        param_op_ini: Union[np.ndarray, None] = None,
        parameter_seed: Union[int, None] = 0,
        caching: bool = True,
    ) -> None:

        super().__init__()

        self.encoding_circuit = encoding_circuit
        self.executor = executor
        self.ml_model = ml_model
        self.ml_model_options = ml_model_options
        self.num_operators = num_operators
        self.operator_seed = operator_seed
        self.operators = operators
        self.param_ini = param_ini
        self.param_op_ini = param_op_ini
        self.parameter_seed = parameter_seed
        self.caching = caching

        self._ml_model = None
        self._initialize_observables()
        self._initialize_lowlevel_qnn()
        self._initialize_ml_model()

        initialize_parameters = (
            self.param_ini is None or len(self.param_ini) != self._qnn.num_parameters
        )
        initialize_parameters_obs = (
            self.param_op_ini is None
            or len(self.param_op_ini) != self._qnn.num_parameters_observable
        )

        self._initialize_parameters(initialize_parameters, initialize_parameters_obs)

    @property
    def used_operators(self) -> List[ObservableBase]:
        """Returns the operators used in the QNN model."""
        return self._operators

    @property
    def qnn(self) -> LowLevelQNN:
        """Returns the underlying low-level QNN object."""
        return self._qnn

    def fit(self, X, y) -> None:
        """Fit a new Quantum Reservoir Computing model to data.

        Args:
            X: Input data
            y: Labels
        """
        X, y = self._validate_input(X, y, incremental=False, reset=False)
        X_qnn = self._qnn.evaluate(X, self.param_ini, self.param_op_ini, "f")["f"]
        self._ml_model.fit(X_qnn, y)

    def predict(self, X) -> np.ndarray:
        """Predict using the Quantum Reservoir Computing.

        Args:
            X : The input data.

        Returns:
            np.ndarray : The predicted values.
        """
        X = self._validate_data(X, accept_sparse=["csr", "csc"], reset=False)
        X_qnn = self._qnn.evaluate(X, self.param_ini, self.param_op_ini, "f")["f"]
        return self._ml_model.predict(X_qnn)

    def _initialize_observables(self) -> None:
        """Create the observables for the QNN model."""

        if isinstance(self.operators, str):
            if self.operators == "random_paulis":
                # Generate random operators
                paulis = random_pauli_list(
                    self.encoding_circuit.num_qubits,
                    self.num_operators,
                    seed=self.operator_seed,
                    phase=False,
                )
                self._operators = [
                    CustomObservable(self.encoding_circuit.num_qubits, str(p)) for p in paulis
                ]
            elif self.operators == "single_paulis":
                # Generate single qubit Pauli operators
                self._operators = []
                for i in range(self.encoding_circuit.num_qubits):
                    for p in ["X", "Y", "Z"]:
                        self._operators.append(SinglePauli(self.encoding_circuit.num_qubits, i, p))
            else:
                raise ValueError("Invalid string for operators")
        else:
            if isinstance(self.operators, ObservableBase):
                self._operators = [self.operators]
            elif isinstance(self.operators, list):
                self._operators = self.operators
            else:
                raise ValueError(
                    "Invalid operators. Must be an ObservableBase object or a list of ObservableBase objects or None."
                )
            self.num_operators = len(self._operators)

    def _initialize_parameters(
        self, parameters: bool = True, parameters_optimizers: bool = True
    ) -> None:
        """
        Initialize the parameters of the QNN model.

        Args:
            parameters (bool): If True, initialize the parameters of the encoding circuit.
            parameters_optimizers (bool): If True, initialize the parameters of the operators
        """
        if parameters:
            self.param_ini = self.encoding_circuit.generate_initial_parameters(
                seed=self.parameter_seed
            )

        if parameters_optimizers:
            self.param_op_ini = np.concatenate(
                [
                    operator.generate_initial_parameters(seed=self.parameter_seed)
                    for operator in self._operators
                ]
            )

    def _initialize_lowlevel_qnn(self) -> None:
        """Initialize the low-level QNN object."""
        self._qnn = LowLevelQNN(
            self.encoding_circuit, self._operators, self.executor, caching=self.caching
        )

    @abstractmethod
    def _initialize_ml_model(self) -> None:
        """Initialize the machine learning model, has to be implemented in the child class"""
        raise NotImplementedError

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
            params.update(self._ml_model.get_params(deep=True))

        return params

    def set_params(self, **params) -> "BaseQRC":
        """
        Sets the hyper-parameters of the QLEM model.

        Args:
            params (dict): A dictionary of hyper-parameters to set.

        Returns:
            BaseQRC: The modified QLEM model.
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

        initialize_lowlevel_qnn = False

        # Set encoding_circuit parameters
        ec_params = params.keys() & self.encoding_circuit.get_params(deep=True).keys()
        if ec_params:
            self.encoding_circuit.set_params(**{key: params[key] for key in ec_params})
            initialize_lowlevel_qnn = True

        # Set qnn parameters
        qnn_params = params.keys() & self._qnn.get_params(deep=True).keys() - ec_params
        if qnn_params:
            self._qnn.set_params(**{key: params[key] for key in qnn_params})
            initialize_lowlevel_qnn = True

        if (
            "num_qubits" in params
            or "num_operators" in params
            or "operator_seed" in params
            or "operators" in params
        ):
            self._initialize_observables()
            initialize_lowlevel_qnn = True
        if "ml_model" in params or "ml_model_options" in params:
            self._initialize_ml_model()

        if initialize_lowlevel_qnn:
            self._initialize_lowlevel_qnn()
            # Reinitialize parameters if the number of parameters has changed
            initialize_parameters = (
                self.param_ini is None or len(self.param_ini) != self._qnn.num_parameters
            )
            initialize_parameters_obs = (
                self.param_op_ini is None
                or len(self.param_op_ini) != self._qnn.num_parameters_observable
            )

            self._initialize_parameters(initialize_parameters, initialize_parameters_obs)

        if "parameter_seed" in params:
            self._initialize_parameters()

        return self

    def _validate_input(self, X, y, incremental, reset):
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            multi_output=True,
            y_numeric=True,
            reset=reset,
        )
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        return X, y
