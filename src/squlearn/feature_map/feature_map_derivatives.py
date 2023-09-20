import numpy as np
from typing import Union, Set

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.compiler import transpile  # TODO

from .feature_map_base import FeatureMapBase

from ..util.optree.optree import (
    OpTreeElementBase,
    OpTreeLeafCircuit,
    OpTreeNodeSum,
    OpTreeNodeList,
    OpTreeNodeBase,
)
from ..util.optree.optree_derivative import simplify_copy, derivative
from ..util.optree.optree_evaluate import optree_assign_parameters
from ..util.data_preprocessing import adjust_input

SUPPORTED_GATES = {"x", "y", "z", "h", "rx", "ry", "rz", "p", "cx", "cy", "cz"}


class FeatureMapDerivatives:
    r"""
    Class for automatic differentiation of feature maps.

    This class allows to compute derivatives of a feature map with respect to its parameters
    by utilizing the parameter-shift rule.
    The derivatives can be obtained by the method :meth:`get_derivative`.
    The type of derivative can be specified by either a string (see table below)
    or a ParameterVector or (a list) of ParameterElements that can be accessed
    via :meth:`feature_vector` or :meth:`parameter_vector`, respectively.

    .. list-table:: Strings that are recognized by the :meth:`get_derivative` method
       :widths: 25 75
       :header-rows: 1

       * - String
         - Derivative
       * - ``"I"``
         - Identity Operation (returns the feature map circuit)
       * - ``"dx"``
         - Gradient with respect to feature :math:`x`:
           :math:`\nabla_x = \big( \frac{\partial}{\partial x_1},\ldots,
           \frac{\partial}{\partial x_n} \big)`
       * - ``"dp"``
         - Gradient with respect to parameter :math:`p`:
           :math:`\nabla_p = \big( \frac{\partial}{\partial p_1},\ldots,
           \frac{\partial}{\partial p_m} \big)`
       * - ``"dxdx"``
         - Hessian with respect to feature :math:`x`:
           :math:`H^x_{ij} = \frac{\partial^2}{\partial x_i \partial x_j}`
       * - ``"dpdxdx"``
         - Derivative of the feature Hessian with respect to parameter :math:`p`:
           :math:`\nabla_p H^x_{ij} = \big( \frac{\partial H^x_{ij}}{\partial p_1},\ldots,
           \frac{\partial H^x_{ij}}{\partial p_m} \big)`
       * - ``laplace``
         - Laplace operator with respect to :math:`x`:
           :math:`\Delta = \nabla^2 = \sum_i \frac{\partial^2}{\partial x^2_i}`
       * - ``laplace_dp``
         - Derivative of the Laplacian with respect to parameter :math:`p`:
           :math:`\nabla_p \circ \Delta = \big( \frac{\partial }{\partial p_1}\Delta,\ldots,
           \frac{\partial}{\partial p_m} \Delta \big)`
       * - ``"dpdp"``
         - Hessian with respect to parameter :math:`p`:
           :math:`H^p_{ij} = \frac{\partial^2}{\partial p_i \partial p_j}`
       * - ``"dxdp"`` (or ``"dxdp"``)
         - Mixed Hessian with respect to feature :math:`x` and parameter :math:`p`:
           :math:`H^{xp}_{ij} = \frac{\partial^2}{\partial x_i \partial p_j}`

    **Example: Feature Map gradient with respect to the trainable parameters**

    .. code-block:: python

       from squlearn.feature_map import QEKFeatureMap, FeatureMapDerivatives
       fm = QEKFeatureMap(num_qubits=2, num_features=2, num_layers=2)
       fm_deriv = FeatureMapDerivatives(fm)
       grad = fm_deriv.get_derivative("dp")

    **Example: Derivative with respect to only the first trainable parameter**

    .. code-block:: python

       from squlearn.feature_map import QEKFeatureMap, FeatureMapDerivatives
       fm = QEKFeatureMap(num_qubits=2, num_features=2, num_layers=2)
       fm_deriv = FeatureMapDerivatives(fm)
       dp0 = fm_deriv.get_derivative((fm_deriv.parameter_vector[0],))


    Args:
        feature_map (FeatureMapBase): Feature map to differentiate
        optree_caching (bool): If True, the optree expressions are cached for faster
                               evaluation. (default: True)
    """

    def __init__(
        self,
        feature_map: FeatureMapBase,
        optree_caching: bool = True,
    ):
        self.feature_map = feature_map

        self._x = ParameterVector("x", self.feature_map.num_features)
        self._p = ParameterVector("p", self.feature_map.num_parameters)

        self._circuit = feature_map.get_circuit(self._x, self._p)

        # TODO: remove -> move to parameter-shift implementation!
        self._instruction_set = list(set(self._circuit.count_ops()))
        self._circuit = _transpile_to_supported_instructions(self._circuit, SUPPORTED_GATES)

        self._optree_start = OpTreeLeafCircuit(self._circuit)

        # self.circuit_optree = CircuitStateFn(primitive=circuit, coeff=1.0)
        self.num_qubits = self._circuit.num_qubits

        self._optree_cache = {}
        self._optree_caching = optree_caching
        if self._optree_caching:
            self._optree_cache["I"] = OpTreeLeafCircuit(self._circuit)

        # get the instruction gates from the initial circuit for transpiling the circuits
        # back to this basis

    def get_derivative(self, derivative: Union[str, tuple]) -> OpTreeElementBase:
        """Determine the derivative of the feature map circuit.

        Args:
            derivative (str or tuple): String or tuple of parameters for specifying the derivation.

        Return:
            Derivative circuit in OpTree format.

        """
        if isinstance(derivative, str):
            if derivative == "I":
                optree = self._optree_start
            elif derivative == "dx":
                optree = self._differentiation_from_tuple((self._x,)).copy()
            elif derivative == "dxdx":
                optree = self._differentiation_from_tuple((self._x, self._x)).copy()
            elif derivative == "dpdxdx":
                optree = self._differentiation_from_tuple((self._p, self._x, self._x)).copy()
            elif derivative == "laplace":
                list_sum = []
                for xi in self._x:
                    list_sum.append(self._differentiation_from_tuple((xi, xi)).copy())
                optree = OpTreeNodeSum(list_sum)
            elif derivative == "laplace_dp":
                list_sum = []
                for xi in self._x:
                    list_sum.append(self._differentiation_from_tuple((self._p, xi, xi)).copy())
                optree = OpTreeNodeSum(list_sum)
            elif derivative == "dp":
                optree = self._differentiation_from_tuple((self._p,)).copy()
            elif derivative == "dpdp":
                optree = self._differentiation_from_tuple((self._p, self._p)).copy()
            elif derivative == "dpdx":
                optree = self._differentiation_from_tuple((self._p, self._x)).copy()
            elif derivative == "dxdp":
                optree = self._differentiation_from_tuple((self._x, self._p)).copy()
            else:
                raise ValueError("Unknown string command:", derivative)
        elif isinstance(derivative, tuple):
            optree = self._differentiation_from_tuple(derivative)
        else:
            raise TypeError("Input is neither string nor tuple, but:", type(derivative))

        return optree

    def get_differentiation_from_tuple(self, diff_tuple: tuple) -> OpTreeElementBase:
        """Returns the derivative of the feature map circuit for a tuple of parameters.

        The tuple describes the differentiation with respect to the parameters in the tuple.

        Args:
            diff_tuple (tuple): Tuple of parameters for specifying the derivation.

        Return:
            Derivative circuit in OpTree format.
        """
        return self.get_derivative(diff_tuple)

    def get_derivation_from_string(self, input_string: str) -> OpTreeElementBase:
        """Returns the derivative of the feature map circuit for a string abbreviation.

        The table for the abbreviations can be found in the documentation of the class.

        Args:
            input_string (str): String for specifying the derivation.

        Return:
            Derivative circuit in OpTree format.
        """
        return self.get_derivative(input_string)

    def _differentiation_from_tuple(self, diff_tuple: tuple) -> OpTreeElementBase:
        """Recursive routine for automatic differentiating the feature map

        Variables for the differentiation are supplied by a tuple
        (x,param,param_op) from left to right -> dx dparam dparam_op PQC(x,param,param_op)


        Args:
            diff_tuple (tuple): tuple containing ParameterVectors or ParameterExpressions or Strings
                                determining the derivation

        Return:
            Derivative circuit in OpTree format.
        """
        # TODO: support tuple of lists of parameter elements

        if diff_tuple == ():
            # Cancel the recursion by returning the optree of the simply measured feature map
            return self._optree_start.copy()
        else:
            # Check if differentiating tuple is already stored in optree_cache
            if self._optree_caching == True and (diff_tuple,) in self._optree_cache:
                # If stored -> return
                return self._optree_cache[(diff_tuple,)].copy()
            else:
                # Recursive differentiation with the most left object
                circ = self._optree_differentiation(
                    self._differentiation_from_tuple(diff_tuple[1:]), diff_tuple[0]
                )
                # Store result in the optree_cache
                if self._optree_caching == True:
                    self._optree_cache[(diff_tuple,)] = circ
                return circ

    @property
    def parameter_vector(self) -> ParameterVector:
        """Parameter ParameterVector ``p`` utilized in the feature map circuit."""
        return self._p

    @property
    def feature_vector(self) -> ParameterVector:
        """Feature ParameterVector ``x`` utilized in the feature map circuit."""
        return self._x

    @property
    def num_parameters(self) -> int:
        """Number of parameters in the feature map circuit."""
        return len(self._p)

    @property
    def num_features(self) -> int:
        """Number of features in the feature map circuit."""
        return len(self._x)

    def assign_parameters(
        self, optree: OpTreeElementBase, features: np.ndarray, parameters: np.ndarray
    ) -> OpTreeElementBase:
        """
        Assigns numerical values to the ParameterVector elements of the feature map circuit.

        Args:
            optree (OperatorBase): OpTree object to be assigned.
            features (np.ndarray): Numerical values of the feature vector.
            parameters (np.ndarray): Numerical values of the parameter vector.

        Return:
            OpTree object with assigned numerical values.
        """

        if optree is None:
            return None

        todo_list = []  # list for the variables
        multi_list = []  # list of return ListOp or no list
        param_list = []  # list of parameters that are substituted

        # check shape of the x and adjust to [[]] form if necessary
        if features is not None:
            xx, multi_x = adjust_input(features, len(self._x))
            todo_list.append(xx)
            param_list.append(self._x)
            multi_list.append(multi_x)
        if parameters is not None:
            pp, multi_p = adjust_input(parameters, len(self._p))
            todo_list.append(pp)
            param_list.append(self._p)
            multi_list.append(multi_p)

        # Recursive construction of the assignment dictionary and list structure
        def rec_assign(dic, todo_list, param_list, multi_list):
            if len(todo_list) <= 0:
                return None
            return_list = []
            for x_ in todo_list[0]:
                for A, B in zip(param_list[0], x_):
                    dic[A] = B
                if len(multi_list[1:]) > 0:
                    return_list.append(
                        rec_assign(dic.copy(), todo_list[1:], param_list[1:], multi_list[1:])
                    )
                else:
                    return_list.append(optree_assign_parameters(optree, dic))

            if multi_list[0]:
                return OpTreeNodeList(return_list)
            else:
                return return_list[0]

        return rec_assign({}, todo_list, param_list, multi_list)

    def _optree_differentiation(
        self,
        optree: OpTreeElementBase,
        parameters: Union[list, tuple, ParameterVectorElement, ParameterVector],
    ) -> OpTreeElementBase:
        """
        Routine for the automatic differentiation based on qiskit routines

        Args:
            optree : Input OpTree expression
            params (list | ParameterVector): variables which are used in the
                                                differentiation (have to be the same type )
        Returns:
            OpTree structure of the differentiated input optree
        """

        # Make list if input is not a list
        if isinstance(parameters, ParameterVectorElement):
            parameters = [parameters]
        if isinstance(parameters, tuple):
            parameters = list(parameters)

        # Call the automatic differentiation routine
        # Separate routine for for one dim. or multi dim. variables
        if len(parameters) == 1:
            return _optree_transpile_back(
                simplify_copy(derivative(optree, parameters).children[0]), self._instruction_set
            )  # TODO:backtranspile maybe in derivative
        else:
            # If multiple variables are differentiated -> results are returned in array

            # Check if the same variables are the same type
            params_name = parameters[0].name.split("[", 1)[0]
            for p in parameters:
                if p.name.split("[", 1)[0] != params_name:
                    raise TypeError("Differentiable variables are not the same type.")

            return _optree_transpile_back(
                simplify_copy(derivative(optree, parameters)), self._instruction_set
            )  # TODO:backtranspile maybe in derivative


def _optree_transpile_back(
    optree_element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit], instruction_set
) -> Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]:
    if isinstance(optree_element, OpTreeNodeBase):
        # Recursive call for all children
        children_list = [
            _optree_transpile_back(child, instruction_set) for child in optree_element.children
        ]
        if isinstance(optree_element, OpTreeNodeSum):
            return OpTreeNodeSum(children_list, optree_element.factor, optree_element.operation)
        elif isinstance(optree_element, OpTreeNodeList):
            return OpTreeNodeList(children_list, optree_element.factor, optree_element.operation)
        else:
            raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")
    elif isinstance(optree_element, (OpTreeLeafCircuit, QuantumCircuit)):
        circuit = optree_element
        if isinstance(optree_element, OpTreeLeafCircuit):
            circuit = optree_element.circuit

        # Transpile back to the given instruction set
        transpiled_circ = transpile(
            circuit,
            basis_gates=instruction_set,
            optimization_level=1,  # 1 for reducing number of gates
            layout_method="trivial",
        )

        if isinstance(optree_element, OpTreeLeafCircuit):
            return OpTreeLeafCircuit(transpiled_circ)

        return transpiled_circ

    else:
        raise ValueError("Unsupported type in _optree_transpile_back:", type(optree_element))


def _transpile_to_supported_instructions(
    circuit: QuantumCircuit, supported_gates: Set[str]
) -> QuantumCircuit:
    """Helper function for transpiling a circuit to a supported instruction set.

    Args:
        circuit (QuantumCircuit): Circuit to transpile.
        supported_gates (Set[str]): Set of supported gates.

    Returns:
        Circuit which is transpiled to the supported instruction set.
    """

    unique_ops = set(circuit.count_ops())
    if not unique_ops.issubset(supported_gates):
        circuit = transpile(
            circuit,
            basis_gates=list(supported_gates),
            optimization_level=0,
            layout_method="trivial",
        )
    return circuit
