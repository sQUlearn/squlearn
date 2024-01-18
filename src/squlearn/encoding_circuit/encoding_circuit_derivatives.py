import numpy as np
from typing import Union, Set

from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from .encoding_circuit_base import EncodingCircuitBase

from ..util.optree.optree import OpTreeElementBase, OpTreeCircuit, OpTreeSum, OpTreeList, OpTree

from ..util.data_preprocessing import adjust_features, adjust_parameters


class EncodingCircuitDerivatives:
    r"""
    Class for automatic differentiation of encoding circuits.

    This class allows to compute derivatives of a encoding circuit with respect to its parameters
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
         - Identity Operation (returns the encoding circuit circuit)
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

    **Example: Encoding Circuit gradient with respect to the trainable parameters**

    .. jupyter-execute::

       from squlearn.encoding_circuit import HubregtsenEncodingCircuit
       from squlearn.encoding_circuit.encoding_circuit_derivatives import EncodingCircuitDerivatives
       fm = HubregtsenEncodingCircuit(num_qubits=2, num_features=2, num_layers=2)
       fm_deriv = EncodingCircuitDerivatives(fm)
       grad = fm_deriv.get_derivative("dp")

    **Example: Derivative with respect to only the first trainable parameter**

    .. jupyter-execute::

       from squlearn.encoding_circuit import HubregtsenEncodingCircuit
       from squlearn.encoding_circuit.encoding_circuit_derivatives import EncodingCircuitDerivatives
       fm = HubregtsenEncodingCircuit(num_qubits=2, num_features=2, num_layers=2)
       fm_deriv = EncodingCircuitDerivatives(fm)
       dp0 = fm_deriv.get_derivative((fm_deriv.parameter_vector[0],))


    Args:
        encoding_circuit (EncodingCircuitBase): Encoding circuit to differentiate
        optree_caching (bool): If True, the OpTree expressions are cached for faster
                               evaluation. (default: True)
    """

    def __init__(
        self,
        encoding_circuit: EncodingCircuitBase,
        optree_caching: bool = True,
    ):
        self.encoding_circuit = encoding_circuit

        self._x = ParameterVector("x", self.encoding_circuit.num_features)
        self._p = ParameterVector("p", self.encoding_circuit.num_parameters)

        self._circuit = encoding_circuit.get_circuit(self._x, self._p)

        self._instruction_set = list(set(self._circuit.count_ops()))
        self._circuit = OpTree.derivative.transpile_to_supported_instructions(self._circuit)

        self._optree_start = OpTreeCircuit(self._circuit)

        # self.circuit_optree = CircuitStateFn(primitive=circuit, coeff=1.0)
        self.num_qubits = self._circuit.num_qubits

        self._optree_cache = {}
        self._optree_caching = optree_caching
        if self._optree_caching:
            self._optree_cache["I"] = OpTreeCircuit(self._circuit)

        # get the instruction gates from the initial circuit for transpiling the circuits
        # back to this basis

    def get_derivative(self, derivative: Union[str, tuple, list]) -> OpTreeElementBase:
        """Determine the derivative of the encoding circuit circuit.

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
                optree = OpTreeSum(list_sum)
            elif derivative == "laplace_dp":
                list_sum = []
                for xi in self._x:
                    list_sum.append(self._differentiation_from_tuple((self._p, xi, xi)).copy())
                optree = OpTreeSum(list_sum)
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
        elif isinstance(derivative, list):
            optree = self._differentiation_from_tuple((derivative,))
        else:
            raise TypeError("Input is neither string nor tuple, but:", type(derivative))

        return optree

    def _differentiation_from_tuple(self, diff_tuple: tuple) -> OpTreeElementBase:
        """Recursive routine for automatic differentiating the encoding circuit

        Variables for the differentiation are supplied by a tuple
        (x,param,param_op) from left to right -> dx dparam dparam_op PQC(x,param,param_op)


        Args:
            diff_tuple (tuple): tuple containing ParameterVectors or ParameterExpressions or Strings
                                determining the derivation

        Return:
            Derivative circuit in OpTree format.
        """

        def helper_hash(diff):
            if isinstance(diff, list):
                return ("list",) + tuple([helper_hash(d) for d in diff])
            elif isinstance(diff, tuple):
                return tuple([helper_hash(d) for d in diff])
            else:
                return diff

        if diff_tuple == ():
            # Cancel the recursion by returning the optree of the simply measured encoding circuit
            return self._optree_start.copy()
        else:
            # Check if differentiating tuple is already stored in optree_cache
            if self._optree_caching == True and helper_hash((diff_tuple,)) in self._optree_cache:
                # If stored -> return
                return self._optree_cache[helper_hash((diff_tuple,))].copy()
            else:
                # Recursive differentiation with the most left object
                circ = self._optree_differentiation(
                    self._differentiation_from_tuple(diff_tuple[1:]), diff_tuple[0]
                )
                # Store result in the optree_cache
                if self._optree_caching == True:
                    self._optree_cache[helper_hash((diff_tuple,))] = circ
                return circ

    @property
    def parameter_vector(self) -> ParameterVector:
        """Parameter ParameterVector ``p`` utilized in the encoding circuit circuit."""
        return self._p

    @property
    def feature_vector(self) -> ParameterVector:
        """Feature ParameterVector ``x`` utilized in the encoding circuit circuit."""
        return self._x

    @property
    def num_parameters(self) -> int:
        """Number of parameters in the encoding circuit circuit."""
        return len(self._p)

    @property
    def num_features(self) -> int:
        """Number of features in the encoding circuit circuit."""
        return len(self._x)

    def assign_parameters(
        self, optree: OpTreeElementBase, features: np.ndarray, parameters: np.ndarray
    ) -> OpTreeElementBase:
        """
        Assigns numerical values to the ParameterVector elements of the encoding circuit circuit.

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
            xx, multi_x = adjust_features(features, len(self._x))
            todo_list.append(xx)
            param_list.append(self._x)
            multi_list.append(multi_x)
        if parameters is not None:
            pp, multi_p = adjust_parameters(parameters, len(self._p))
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
                    return_list.append(OpTree.assign_parameters(optree, dic))

            if multi_list[0]:
                return OpTreeList(return_list)
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

        # If no parameters are given -> return empty list
        if len(parameters) == 0:
            return OpTreeList([])

        # Call the automatic differentiation routine
        # Check if the same variables are the same type
        params_name = parameters[0].name.split("[", 1)[0]
        for p in parameters:
            if p.name.split("[", 1)[0] != params_name:
                raise TypeError("Differentiable variables are not the same type.")

        return OpTree.simplify(OpTree.derivative.differentiate(optree, parameters))
