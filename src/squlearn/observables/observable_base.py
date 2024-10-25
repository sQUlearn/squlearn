import numpy as np
from typing import Union
from abc import ABC, abstractmethod

from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Pauli

from ..util.optree.optree import OpTreeNodeBase, OpTreeList, OpTreeSum, OpTreeOperator, OpTree


class ObservableBase(ABC):
    """Base class for observables.

    Args:
        num_qubits (int): Number of qubits.

    Attributes:
    -----------

    Attributes:
        num_parameters (int): Number of trainable parameters in the observable.
        num_qubits (int): Number of qubits in the observable.

    Methods:
    --------
    """

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits
        self._num_all_qubits = num_qubits
        self._qubit_map = np.linspace(0, num_qubits - 1, num_qubits, dtype=int)
        self._is_mapped = False

    def set_map(self, qubit_map: Union[list, dict], num_all_qubits: int):
        """
        Function for setting a qubit mapping from physical qubits to the ones of the operator.

        This function is necessary whenever the number of physical qubits are different from the
        operator definition, as for example when running on a real backend.
        The number of qubits in the system has to be larger than the number
        of qubits in the observable.

        Args:
            qubit_map (Union[list, dict]): A list or dictionary specifying which of the input
                                           qubits are mapped to the output qubits.
            num_all_qubits (int): The total number of qubits in the system.
        """

        self._qubit_map = qubit_map
        self._num_all_qubits = num_all_qubits
        self._is_mapped = True
        if self._num_all_qubits < self._num_qubits:
            raise ValueError(
                """Number of qubits in the system is smaller than the number
                                of qubits in the observable."""
            )

    @property
    def is_mapped(self):
        """Returns True if the operator is mapped to physical qubits."""
        return self._is_mapped

    @property
    def num_parameters(self):
        """Number of free parameters in the observable."""
        return 0

    @property
    def parameter_bounds(self):
        """Bounds of the free parameters in the observable."""
        return np.array([[0, 5]] * self.num_parameters)

    def generate_initial_parameters(
        self, ones: bool = True, seed: Union[int, None] = None
    ) -> np.ndarray:
        """
        Generates random parameters for the observable

        Args:
            ones (bool): If True, returns an array of ones (default: True)
            seed (Union[int,None]): Seed for the random number generator

        Return:
            The randomly generated parameters
        """
        if ones:
            return np.ones(self.num_parameters)

        if self.num_parameters == 0:
            return np.array([])
        r = np.random.RandomState(seed)
        bounds = self.parameter_bounds
        return r.uniform(low=bounds[:, 0], high=bounds[:, 1])

    @property
    def num_qubits(self):
        """Number of qubits in the observable."""
        return self._num_qubits

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the operator.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        param = {}
        param["num_qubits"] = self._num_qubits
        return param

    def set_params(self, **params) -> None:
        """
        Sets value of the operator hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. ``num_qubits=2``.
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )
            try:
                setattr(self, key, value)
            except:
                setattr(self, "_" + key, value)

            # Reset Mapping if the number of quibts is changed
            if key == "num_qubits" or key == "_num_qubits":
                self._num_all_qubits = self._num_qubits
                self._qubit_map = np.linspace(0, self._num_qubits - 1, self._num_qubits, dtype=int)

    def get_operator(self, parameters: Union[ParameterVector, np.ndarray]) -> SparsePauliOp:
        """Returns Operator as a SparsePauliOp expression.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Vector of parameters used in
                                                             the operator
        Return:
            StateFn expression of the observable.
        """

        if self._is_mapped:
            return self.get_pauli_mapped(parameters)
        else:
            return self.get_pauli(parameters)

    @abstractmethod
    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]) -> SparsePauliOp:
        """
        Returns the PauliOp expression of the observable.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Vector of parameters used
                                                             in the operator

        Returns:
            Expectation operator in Qiskit's PauliOp class
        """
        raise NotImplementedError

    def get_pauli_mapped(self, parameters: Union[ParameterVector, np.ndarray]) -> SparsePauliOp:
        """
        Changes the operator to the physical qubits, set by :meth:`set_map`.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Vector of parameters used in
                                                             the operator

        Return:
            Expectation operator in Qiskit's SprasePauliOp class with qubits mapped to
            physical ones
        """

        def map_operator(operator):
            if isinstance(operator, OpTreeNodeBase):
                children_list = [map_operator(op) for op in operator.children]

                # Rebuild the tree with the new children and factors (copy part)
                if isinstance(operator, OpTreeSum):
                    return OpTreeSum(children_list, operator.factor, operator.operation)
                elif isinstance(operator, OpTreeList):
                    return OpTreeList(children_list, operator.factor, operator.operation)
                else:
                    raise ValueError("Wrong Type in operator: ", type(operator))
            elif isinstance(operator, (SparsePauliOp, OpTreeOperator)):
                op = operator
                if isinstance(op, OpTreeOperator):
                    op = op.operator

                op_list = []
                for op_ in op.paulis:
                    blank = Pauli("I" * self._num_all_qubits)
                    for i, p in enumerate(op_):
                        blank[self._qubit_map[i]] = p
                    op_list.append(blank)
                return SparsePauliOp(op_list, op.coeffs)

            else:
                raise ValueError("Wrong Type in operator: ", type(operator))

        return map_operator(self.get_pauli(parameters))

    def __str__(self):
        """Return a string representation of the ObservableBase."""
        p = ParameterVector("p", self.num_parameters)
        return str(self.get_operator(p))

    def __repr__(self):
        """Return a string representation of the ObservableBase."""
        p = ParameterVector("p", self.num_parameters)
        return repr(self.get_operator(p))

    def __add__(self, x):
        """Addition of two observables."""
        if not isinstance(x, ObservableBase):
            raise ValueError("Only the addition with other observable is allowed!")

        class AddedObservable(ObservableBase):
            """Internal class for a sum of observables.

            Args:
                op1 (ObservableBase): Left observable
                op2 (ObservableBase): Right observable
            """

            def __init__(self, op1: ObservableBase, op2: ObservableBase):
                if op1.num_qubits != op2.num_qubits:
                    raise ValueError("Number of qubits is not equal in both observable.")

                super().__init__(op1.num_qubits)

                self._op1 = op1
                self._op2 = op2

            def get_params(self, deep: bool = True) -> dict:
                """
                Returns hyper-parameters and their values of the composed encoding circuit.

                Hyper-parameter names are prefixed by ``op1__`` or ``op2__`` depending on
                which encoding circuit they belong to.

                Args:
                    deep (bool): If True, also the parameters for
                                 contained objects are returned (default=True).

                Return:
                    Dictionary with hyper-parameters and values.
                """

                if self._op1 == self._op2:
                    return self._op1.get_params()
                else:
                    params = dict(op1=self._op1, op2=self._op2)
                    if deep:
                        deep_items = self._op1.get_params().items()
                        for k, val in deep_items:
                            if k != "num_qubits":
                                params["op1__" + k] = val
                        deep_items = self._op2.get_params().items()
                        for k, val in deep_items:
                            if k != "num_qubits":
                                params["op2__" + k] = val

                    params["num_qubits"] = self._op1.get_params()["num_qubits"]

                    return params

            def set_params(self, **params) -> None:
                """
                Sets value of the composed kernel hyper-parameters.

                Args:
                    params: Hyper-parameters and their values, e.g. ``num_qubits=2``
                """
                valid_params = self.get_params()
                op1_dict = {}
                op2_dict = {}
                for key, value in params.items():
                    if key not in valid_params:
                        raise ValueError(
                            f"Invalid parameter {key!r}. "
                            f"Valid parameters are {sorted(valid_params)!r}."
                        )

                    if self._op1 == self._op2:
                        op1_dict[key] = value
                    else:
                        if key.startswith("op1__"):
                            op1_dict[key[5:]] = value
                        elif key.startswith("op2__"):
                            op2_dict[key[5:]] = value

                        if key == "num_qubits":
                            op1_dict["num_qubits"] = value
                            op2_dict["num_qubits"] = value

                if len(op1_dict) > 0:
                    self._op1.set_params(**op1_dict)
                if len(op2_dict) > 0:
                    self._op2.set_params(**op2_dict)

            @property
            def num_parameters(self) -> int:
                """The number of trainable parameters of added observable.

                Is equal to the sum of both trainable parameters.
                """
                if self._op1 == self._op2:
                    return self._op1.num_parameters
                else:
                    return self._op1.num_parameters + self._op2.num_parameters

            @property
            def parameter_bounds(self) -> np.ndarray:
                """The bounds of the trainable parameters of added observable."""
                if self._op1 == self._op2:
                    return self._op1.parameter_bounds
                else:
                    return np.concatenate(
                        (self._op1.parameter_bounds, self._op2.parameter_bounds), axis=0
                    )

            def generate_initial_parameters(
                self, ones: bool = True, seed: Union[int, None] = None
            ) -> np.ndarray:
                """
                Generates random parameters for the observable.

                Args:
                    ones (bool): If True, returns an array of ones (default: True)
                    seed (Union[int,None]): Seed for the random number generator (default: None)

                Return:
                    The randomly generated parameters
                """
                return np.concatenate(
                    (
                        self._op1.generate_initial_parameters(ones, seed),
                        self._op2.generate_initial_parameters(ones, seed),
                    ),
                    axis=0,
                )

            def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]):
                """Returns the PauliOp expression of the added observable.

                Args:
                    parameters (Union[ParameterVector, np.ndarray]): Vector of parameters used
                                                                     in the operator

                Return:
                    PauliOp: Expectation operator in Qiskit's PauliOp class
                """
                if self._op1 == self._op2:
                    paulis_op = self._op1.get_pauli(parameters)
                    return OpTree.simplify(paulis_op + paulis_op)
                else:
                    paulis_op1 = self._op1.get_pauli(parameters[: self._op1.num_parameters])
                    paulis_op2 = self._op2.get_pauli(parameters[self._op1.num_parameters :])
                    return OpTree.simplify(paulis_op1 + paulis_op2)

        return AddedObservable(self, x)

    def __mul__(self, x):
        """Multiplication of two observables."""
        if not isinstance(x, ObservableBase):
            raise ValueError("Only the multiplication with other observable is allowed!")

        class MultipliedObservable(ObservableBase):
            """Internal class for a multiplication of observables.

            Args:
                op1 (ObservableBase): Left observable
                op2 (ObservableBase): Right observable
            """

            def __init__(self, op1: ObservableBase, op2: ObservableBase):
                if op1.num_qubits != op2.num_qubits:
                    raise ValueError("Number of qubits is not equal in both observable.")

                super().__init__(op1.num_qubits)

                self._op1 = op1
                self._op2 = op2

            def get_params(self, deep: bool = True) -> dict:
                """
                Returns hyper-parameters and their values of the composed encoding circuit.

                Hyper-parameter names are prefixed by ``op1__`` or ``op2__`` depending on
                which encoding circuit they belong to.

                Args:
                    deep (bool): If True, also the parameters for
                                 contained objects are returned (default=True).

                Return:
                    Dictionary with hyper-parameters and values.
                """
                if self._op1 == self._op2:
                    return self._op1.get_params()
                else:
                    params = dict(op1=self._op1, op2=self._op2)
                    if deep:
                        deep_items = self._op1.get_params().items()
                        for k, val in deep_items:
                            if k != "num_qubits":
                                params["op1__" + k] = val
                        deep_items = self._op2.get_params().items()
                        for k, val in deep_items:
                            if k != "num_qubits":
                                params["op2__" + k] = val

                    params["num_qubits"] = self._op1.get_params()["num_qubits"]

                    return params

            def set_params(self, **params) -> None:
                """
                Sets value of the composed kernel hyper-parameters.

                Args:
                    params: Hyper-parameters and their values, e.g. ``num_qubits=2``
                """
                valid_params = self.get_params()
                op1_dict = {}
                op2_dict = {}
                for key, value in params.items():
                    if key not in valid_params:
                        raise ValueError(
                            f"Invalid parameter {key!r}. "
                            f"Valid parameters are {sorted(valid_params)!r}."
                        )

                    if self._op1 == self._op2:
                        op1_dict[key] = value
                    else:
                        if key.startswith("op1__"):
                            op1_dict[key[5:]] = value
                        elif key.startswith("op2__"):
                            op2_dict[key[5:]] = value

                        if key == "num_qubits":
                            op1_dict["num_qubits"] = value
                            op2_dict["num_qubits"] = value

                if len(op1_dict) > 0:
                    self._op1.set_params(**op1_dict)
                if len(op2_dict) > 0:
                    self._op2.set_params(**op2_dict)

            @property
            def num_parameters(self) -> int:
                """The number of trainable parameters of multiplied observable.

                Is equal to the sum of both trainable parameters.
                """
                if self._op1 == self._op2:
                    return self._op1.num_parameters
                else:
                    return self._op1.num_parameters + self._op2.num_parameters

            def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]):
                """Returns the PauliOp expression of the multiplied observable.

                Args:
                    parameters (Union[ParameterVector, np.ndarray]): Vector of parameters used
                                                                     in the operator

                Return:
                    PauliOp: Expectation operator in Qiskit's PauliOp class
                """
                if self._op1 == self._op2:
                    paulis_op = self._op1.get_pauli(parameters)
                    return OpTree.simplify(paulis_op.compose(paulis_op))
                else:
                    paulis_op1 = self._op1.get_pauli(parameters[: self._op1.num_parameters])
                    paulis_op2 = self._op2.get_pauli(parameters[self._op1.num_parameters :])
                    return OpTree.simplify(paulis_op1.compose(paulis_op2))

        return MultipliedObservable(self, x)
