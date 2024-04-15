from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.quantum_info import SparsePauliOp
from typing import Union
import numpy as np

from .observable_base import ObservableBase
from ..util.data_preprocessing import adjust_parameters

from ..util.optree.optree import (
    OpTreeElementBase,
    OpTreeSum,
    OpTreeList,
    OpTreeOperator,
    OpTree,
)


class ObservableDerivatives:
    r"""Class for calculating derivatives of observables.

    The derivatives are calculated by automatic differentiation of parameter in the expectation
    operator. Also, squaring of the operator is implemented.
    The class can either applied on a single operator, or on a list of operators.
    results are cached for faster evaluation.

    Args:
        observable (Union[ObservableBase, list]): Expectation operator or list
                                                                     of observables from
                                                                     which the derivatives are
                                                                     obtained.
        optree_caching (bool): If True, the optree structure of the observable is cached

    .. list-table:: Strings that are recognized by the :meth:`get_derivative` method
       :widths: 25 75
       :header-rows: 1

       * - String
         - Derivative
       * - ``"O"``
         - Expectation operator :math:`\hat{O}`
       * - ``"OO"``
         - Squared observable :math:`\hat{O}^2`
       * - ``"dop"`` or ``"Odop"``
         - First-order derivative of the observable:
           :math:`\frac{d}{dp}\hat{O}(p)`
       * - ``"dopdop"`` or ``"Odopdop"``
         - Second-order derivative of the observable:
           :math:`\frac{d^2}{dp^2}\hat{O}(p)`
       * - ``"OOdop"``
         - First-order derivative of the squared observable:
           :math:`\frac{d}{dp}\hat{O}^2(p)`
       * - ``"OOdopdop"``
         - Second-order derivative of the squared observable:
           :math:`\frac{d^2}{dp^2}\hat{O}^2(p)`
       * - ``"I"``
         - Returns a identity operator with the same number of qubits as the provided
           observable

    **Example: first-order derivative of the Ising Hamiltonian**

    .. jupyter-execute::

       from squlearn.observables import IsingHamiltonian
       from squlearn.observables.observable_derivatives import ObservableDerivatives
       op = IsingHamiltonian(num_qubits=3)
       print(ObservableDerivatives(op).get_derivative("dop"))

    **Example: Squared summed Pauli Operator**

    .. jupyter-execute::

       from squlearn.observables import SummedPaulis
       from squlearn.observables.observable_derivatives import ObservableDerivatives
       op = SummedPaulis(num_qubits=3)
       print(ObservableDerivatives(op).get_operator_squared())

    Attributes:
    -----------

    Attributes:
        parameter_vector (ParameterVector): Parameter vector used in the observable
        num_parameters (int): Total number of trainable parameters in the observable
        num_operators (int): Number operators in case of multiple observables

    Methods:
    --------
    """

    def __init__(
        self,
        observable: Union[ObservableBase, list],
        optree_caching=True,
        split_paulis=False,
    ):
        self._observable = observable
        self._split_paulis = split_paulis

        # Contains the OperatorMeasurement() of the expectation-operator for later replacement
        if isinstance(self._observable, ObservableBase):
            # 1d output by a single expectation-operator
            self.multiple_output = False
            self._num_operators = 1
            self._parameter_vector = ParameterVector("p_op", observable.num_parameters)
            optree = OpTreeOperator(self._observable.get_operator(self._parameter_vector))
        else:
            # multi dimensional output by multiple Expectation-operators
            observable_list = []
            self.multiple_output = True
            self._num_operators = len(observable)
            try:
                n_oper = 0
                for op in self._observable:
                    n_oper = n_oper + op.num_parameters

                self._parameter_vector = ParameterVector("p_op", n_oper)
                ioff = 0
                for op in self._observable:
                    observable_list.append(
                        OpTreeOperator(op.get_operator(self._parameter_vector[ioff:]))
                    )
                    ioff = ioff + op.num_parameters
                optree = OpTreeList(observable_list)
            except:
                raise ValueError("Unknown structure of the Expectation operator!")

        self._optree_start = optree
        self._optree_cache = {}
        self._optree_caching = optree_caching

    def get_derivative(self, derivative: Union[str, tuple, list]) -> OpTreeElementBase:
        """Determine the derivative of the observable.

        Args:
            derivative (str or tuple): String or tuple of parameters for specifying the derivation.
                                       See :class:`ObservableDerivatives` for more
                                       information.

        Return:
            Differentiated observable in OpTree format
        """
        if isinstance(derivative, str):
            # todo change with replaced operator
            if derivative == "I":
                measure_op = OpTreeOperator(SparsePauliOp("I" * self._observable.num_qubits))
            elif derivative == "O":
                measure_op = self.get_operator()
            elif derivative == "OO":
                measure_op = self.get_operator_squared()
            elif derivative == "dop" or derivative == "Odop":
                measure_op = self._differentiation_from_tuple(
                    self._optree_start.copy(), (self._parameter_vector,), "O"
                )
            elif derivative == "dopdop" or derivative == "Odopdop":
                measure_op = self._differentiation_from_tuple(
                    self._optree_start.copy(),
                    (self._parameter_vector, self._parameter_vector),
                    "O",
                )
            elif derivative == "OOdop":
                measure_op = self._differentiation_from_tuple(
                    self.get_operator_squared(), (self._parameter_vector,), "OO"
                )
            elif derivative == "OOdopdop":
                measure_op = self._differentiation_from_tuple(
                    self.get_operator_squared(),
                    (self._parameter_vector, self._parameter_vector),
                    "OO",
                )
            else:
                raise ValueError("Unknown string command:", derivative)

        elif isinstance(derivative, tuple):
            measure_op = self._differentiation_from_tuple(self._optree_start, derivative, "O")
        elif isinstance(derivative, list):
            measure_op = self._differentiation_from_tuple(self._optree_start, (derivative,), "O")
        else:
            raise TypeError("Input is neither string, list nor tuple, but:", type(derivative))

        measure_op.replace = False
        return measure_op

    def _differentiation_from_tuple(
        self, optree: OpTreeElementBase, diff_tuple: tuple, observable_label: str
    ) -> OpTreeElementBase:
        """Recursive routine for automatic differentiating the observable

        Args:
            optree (OpTreeElementBase): optree structure of the observable
            diff_tuple (tuple): Tuple containing ParameterVectors or ParameterExpressions
            observable_label (str): string for labeling the observable

        Return:
            The differentiated OpTree expression
        """

        def helper_hash(diff):
            if isinstance(diff, list):
                return ("list",) + tuple([helper_hash(d) for d in diff])
            elif isinstance(diff, tuple):
                return tuple([helper_hash(d) for d in diff])
            else:
                return diff

        if diff_tuple == ():
            # Cancel the recursion by returning the optree operator
            return optree
        else:
            # Check if differentiating tuple is already stored in optree_cache
            if (
                self._optree_caching == True
                and (helper_hash(diff_tuple), observable_label) in self._optree_cache
            ):
                # If stored -> return
                return self._optree_cache[(diff_tuple, observable_label)].copy()
            else:
                # Recursive differentiation with the most left object
                measure = operator_differentiation(
                    self._differentiation_from_tuple(optree, diff_tuple[1:], observable_label),
                    diff_tuple[0],
                )

                if self._split_paulis:
                    measure = OpTree.evaluate.transform_to_zbasis(measure)

                # Store result in the optree_cache
                if self._optree_caching == True:
                    self._optree_cache[(helper_hash(diff_tuple), observable_label)] = measure
                return measure

    def get_operator(self):
        "Returns the observable operator"
        if self._optree_caching == True and "O" in self._optree_cache:
            return self._optree_cache["O"].copy()
        else:
            O = self._optree_start
            if self._split_paulis:
                O = OpTree.evaluate.transform_to_zbasis(O)
            # If caching is enabled, store in the dictionary
            if self._optree_caching == True:
                self._optree_cache["O"] = O
            return O

    def get_operator_squared(self):
        "Returns the squared form of the observable OO=O^2"
        if self._optree_caching == True and "OO" in self._optree_cache:
            return self._optree_cache["OO"].copy()
        else:

            def recursive_squaring(op):
                if isinstance(op, OpTreeOperator):
                    return OpTreeOperator(op.operator.power(2))
                elif isinstance(op, SparsePauliOp):
                    return op.operator.power(2)
                elif isinstance(op, OpTreeSum):
                    return OpTreeSum(
                        [recursive_squaring(child) for child in op.children],
                        op.factor,
                        op.operation,
                    )
                elif isinstance(op, OpTreeList):
                    return OpTreeList(
                        [recursive_squaring(child) for child in op.children],
                        op.factor,
                        op.operation,
                    )
                else:
                    raise ValueError("Unknown type in recursive_squaring:", type(op))

            O2 = OpTree.simplify(recursive_squaring(self._optree_start))
            if self._split_paulis:
                O2 = OpTree.evaluate.transform_to_zbasis(O2)

            # If caching is enabled, store in the dictionary
            if self._optree_caching == True:
                self._optree_cache["OO"] = O2

            return O2

    @property
    def parameter_vector(self):
        """Parameter vector of the observable"""
        return self._parameter_vector

    @property
    def num_parameters(self):
        """Total number of trainable parameters in the observable"""
        return len(self._parameter_vector)

    @property
    def num_operators(self):
        """Number operators in case of multiple observables"""
        return self._num_operators

    def assign_parameters(
        self, operator: OpTreeElementBase, parameters: np.ndarray
    ) -> OpTreeElementBase:
        """Assign parameters to a derivative that is obtained from this class.

        Args:
            operator (OperatorBase): Operator to which the parameters are assigned
            parameters (np.ndarray): Parameters values that replace the parameters in the operator

        Return:
            Operator with assigned parameters
        """
        param_op_inp, multi_param_op = adjust_parameters(parameters, len(self._parameter_vector))

        return_list = []
        for p in param_op_inp:
            dic = dict(zip(self._parameter_vector, p))
            return_list.append(OpTree.assign_parameters(operator, dic))

        if multi_param_op:
            return OpTreeList(return_list)
        else:
            return return_list[0]


def operator_differentiation(
    optree: OpTreeElementBase, parameters: Union[ParameterVector, list, ParameterExpression]
) -> OpTreeElementBase:
    """Function for differentiating a given observable w.r.t. to its parameters

    Args:
        optree (OpTreeElementBase): optree structure of the observable, can also be a
                                            list of observables
        parameters: Union[ParameterVector, list, ParameterExpression]: Parameters that are used for
                                                                       the differentiation.
    Returns:
        Differentiated observable as an OpTree
    """
    # Make a list if input is not a list
    if parameters == None or parameters == []:
        return None

    if isinstance(parameters, ParameterVectorElement):
        parameters = [parameters]

    # If no parameters are given -> return empty list
    if len(parameters) == 0:
        return OpTreeList([])

    # Check if the same variables are the same type
    params_name = parameters[0].name.split("[", 1)[0]
    for p in parameters:
        if p.name.split("[", 1)[0] != params_name:
            raise TypeError("Differentiable variables are not the same type.")

    return OpTree.simplify(OpTree.derivative.differentiate(optree, parameters))
