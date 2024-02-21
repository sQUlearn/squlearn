import abc
from typing import Union
import numpy as np
import copy

from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

import pennylane as qml
import pennylane.numpy as pnp

from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ..util import Executor

from ..util.pennylane import PennyLaneCircuit, PennyLaneDevice

from .lowlevel_qnn_base import LowLevelQNNBase

from ..util.data_preprocessing import adjust_features, adjust_parameters, to_tuple

# import tensorflow as tf

# import jax
# import jax.numpy as jnp
# from jax.config import config
# config.update("jax_enable_x64", True)

# import torch


class direct_evaluation:
    """ Class for evaluation of derivatives of the QNN """
    def __init__(
        self,
        key,
        order=0,
        argnum=None,
        return_grad_param=False,
        return_grad_param_obs=False,
        return_grad_x=False,
        squared=False,
    ):
        self.key = key
        self.order = order
        self.argnum = argnum
        self.return_grad_param = return_grad_param
        self.return_grad_param_obs = return_grad_param_obs
        self.return_grad_x = return_grad_x
        self.squared = squared


class post_processing_evaluation:
    """ Class post processing evaluation of derivatives of the QNN"""
    def __init__(self, key, evaluation_tuple, evaluation_function):
        self.key = key
        self.evaluation_tuple = evaluation_tuple
        self.evaluation_function = evaluation_function


def _get_class_from_string(val: str):
    """Converts an input string to the Expec data structure.

    Args:
        String that defines the expectation value derivative

    Returns:
        Associated Expec object

    """

    def eval_var(value_dict: dict):
        """Evaluate the variance of the QNN output

        Args:
            value_dict (dict): Dictionary of QNN derivatives values

        Returns:
            Variance of the QNN output
        """
        return value_dict["fcc"] - np.square(value_dict["f"])

    def eval_dvardx(value_dict: dict):
        """Evaluate the derivative of the variance with respect to the input

        Args:
            value_dict (dict): Dictionary of QNN derivatives values

        Returns:
            Derivative of the variance with respect to the input
        """
        return_val = np.zeros(value_dict["dfccdx"].shape)
        for i in range(value_dict["dfccdx"].shape[-1]):
            return_val[..., i] = value_dict["dfccdx"][..., i] - 2.0 * (
                np.multiply(
                    value_dict["dfdx"][..., i],
                    value_dict["f"],
                )
            )
        return return_val

    def eval_dvardp(value_dict):
        """Evaluate the derivative of the variance with respect to the parameters

        Args:
            value_dict (dict): Dictionary of QNN derivatives values

        Returns:
            Derivative of the variance with respect to the parameters
        """
        return_val = np.zeros(value_dict["dfccdp"].shape)
        for i in range(value_dict["dfccdp"].shape[-1]):
            return_val[..., i] = value_dict["dfccdp"][..., i] - 2.0 * (
                np.multiply(
                    value_dict["dfdp"][..., i],
                    value_dict["f"],
                )
            )
        return return_val

    def eval_dvardop(value_dict):
        """Evaluate the derivative of the variance with respect to the observable parameters

        Args:
            value_dict (dict): Dictionary of QNN derivatives values

        Returns:
            Derivative of the variance with respect to the observable parameters
        """
        return_val = np.zeros(value_dict["dfccdop"].shape)
        for i in range(value_dict["dfccdop"].shape[-1]):
            return_val[..., i] = value_dict["dfccdop"][..., i] - 2.0 * (
                np.multiply(
                    value_dict["dfdop"][..., i],
                    value_dict["f"],
                )
            )
        return return_val

    def get_eval_laplace(todo: str):

        def eval_laplace(value_dict):
            return_val = np.zeros(value_dict[todo].shape[:-2])
            for i in range(value_dict[todo].shape[-1]):
                return_val+=value_dict[todo][...,i,i]
            return return_val

        return eval_laplace

    if isinstance(val, str):
        if val == "f":
            return direct_evaluation("f")
        elif val == "dfdx":
            return direct_evaluation("dfdx", 1, argnum=[1], return_grad_x=True)
        elif val == "dfdxdx":
            return direct_evaluation("dfdxdx", 2, argnum=[1, 1], return_grad_x=True)
        elif val == "laplace":
            return post_processing_evaluation(val,("dfdxdx",), get_eval_laplace("dfdxdx"))
        elif val == "laplace_dp":
            return post_processing_evaluation(val,("dfdpdxdx",), get_eval_laplace("dfdpdxdx"))
        elif val == "laplace_dop":
            return post_processing_evaluation(val,("dfdopdxdx",), get_eval_laplace("dfdopdxdx"))
        elif val == "dfdp":
            return direct_evaluation("dfdp", 1, argnum=[0], return_grad_param=True)
        elif val == "dfdpdp":
            return direct_evaluation("dfdpdp", 2, argnum=[0, 0], return_grad_param=True)
        elif val == "dfdopdp":
            return direct_evaluation(
                "dfdopdp", 2, argnum=[2, 0], return_grad_param=True, return_grad_param_obs=True
            )
        elif val == "dfdpdop":
            return direct_evaluation(
                "dfdpdop", 2, argnum=[0, 2], return_grad_param=True, return_grad_param_obs=True
            )
        elif val == "dfdop":
            return direct_evaluation("dfdop", 1, argnum=[2], return_grad_param_obs=True)
        elif val == "dfdopdop":
            return direct_evaluation("dfdopdop", 2, argnum=[2, 2], return_grad_param_obs=True)
        elif val == "dfdpdx":
            return direct_evaluation("dfdpdx", 2, argnum=[0, 1], return_grad_param=True, return_grad_x=True)
        elif val == "dfdxdp":
            return direct_evaluation("dfdxdp", 2, argnum=[1, 0], return_grad_param=True, return_grad_x=True)
        elif val == "dfdxdxdp":
            return direct_evaluation(
                "dfdxdxdp", 3, argnum=[1, 1, 0], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdxdpdx":
            return direct_evaluation(
                "dfdxdpdx", 3, argnum=[1, 0, 1], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdpdxdx":
            return direct_evaluation(
                "dfdpdxdx", 3, argnum=[0, 1, 1], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdopdx":
            return direct_evaluation(
                "dfdopdx", 2, argnum=[2, 1], return_grad_param_obs=True, return_grad_x=True
            )
        elif val == "dfdopdxdx":
            return direct_evaluation(
                "dfdopdxdx",
                3,
                argnum=[2, 1, 1],
                return_grad_param_obs=True,
                return_grad_x=True,
            )
        elif val == "fcc":
            return direct_evaluation("fcc", squared=True)
        elif val == "dfccdx":
            return direct_evaluation("dfccdx", 1, argnum=[1], return_grad_x=True, squared=True)
        elif val == "dfccdxdx":
            return direct_evaluation("dfccdxdx", 2, argnum=[1, 1], return_grad_x=True, squared=True)
        elif val == "dfccdp":
            return direct_evaluation("dfccdp", 1, argnum=[0], return_grad_param=True, squared=True)
        elif val == "dfccdpdp":
            return direct_evaluation("dfccdpdp", 2, argnum=[0, 0], return_grad_param=True, squared=True)
        elif val == "dfccdopdx":
            return direct_evaluation(
                "dfccdopdx",
                2,
                argnum=[2, 1],
                return_grad_param_obs=True,
                return_grad_x=True,
                squared=True,
            )
        elif val == "dfccdop":
            return direct_evaluation("dfccdop", 1, argnum=[2], return_grad_param_obs=True, squared=True)
        elif val == "dfccdopdop":
            return direct_evaluation(
                "dfccdopdop", 2, argnum=[2, 2], return_grad_param_obs=True, squared=True
            )
        elif val in ("var", "varf"):
            return post_processing_evaluation(val,("f", "fcc"), eval_var)
        elif val in ("dvardx", "dvarfdx"):
            return post_processing_evaluation(val,("f", "dfccdx", "dfdx"), eval_dvardx)
        elif val in ("dvardp", "dvarfdp"):
            return post_processing_evaluation(val,("f", "dfccdp", "dfdp"), eval_dvardp)
        elif val in ("dvardop", "dvarfdop"):
            return post_processing_evaluation(val,("f", "dfccdop", "dfdop"), eval_dvardop)
        elif val == "fischer":
            return None
        else:
            raise ValueError("Unknown input string:", val)

    elif isinstance(val, ParameterVectorElement):
        raise ValueError("ParameterVectorElement not supported")
    elif isinstance(val, ParameterVector):
        raise ValueError("ParameterVector not supported")
    else:
        raise TypeError("String expected, found type:", type(val))


class LowLevelQNNPennyLane(LowLevelQNNBase):

    def __init__(
        self,
        pqc: EncodingCircuitBase,
        observable: Union[ObservableBase, list],
        executor: Executor,
        result_caching: bool = False,
    ) -> None:

        super().__init__(pqc, observable, executor)

        self._device = PennyLaneDevice()

        self._x = ParameterVector("x", self._pqc.num_features)
        self._param = ParameterVector("param", self._pqc.num_parameters)
        self._qiskit_circuit = self._pqc.get_circuit(self._x, self._param)

        if isinstance(self._observable, ObservableBase):

            self._multiple_output = False
            self._num_operators = 1
            self._num_parameters_observable = self._observable.num_parameters
            self._param_obs = ParameterVector("param_obs", self._num_parameters_observable)
            self._qiskit_observable = self._observable.get_operator(self._param_obs)
            self._qiskit_observable_squared = self._qiskit_observable.power(2).simplify()
        elif isinstance(self._observable, list):
            self._multiple_output = True
            self._num_operators = len(observable)
            self._num_parameters_observable = 0
            for obs in self._observable:
                self._num_parameters_observable += obs.num_parameters
            self._param_obs = ParameterVector("param_obs", self._num_parameters_observable)
            self._qiskit_observable = []
            self._qiskit_observable_squared = []
            ioff = 0
            for obs in self._observable:
                self._qiskit_observable.append(obs.get_operator(self._param_obs[ioff:]))
                self._qiskit_observable_squared.append(self._qiskit_observable[-1].power(2).simplify())
                ioff = ioff + obs.num_parameters
        else:
            raise ValueError("Observable must be of type ObservableBase or list")

        self._pennylane_circuit = PennyLaneCircuit(
            self._device, self._qiskit_circuit, self._qiskit_observable
        )
        self._pennylane_circuit_squared = PennyLaneCircuit(
            self._device, self._qiskit_circuit, self._qiskit_observable_squared
        )

        # Initialize result cache
        self._result_caching = result_caching
        self.result_container = {}

        self._jax_cache = {}

    def draw(self, **kwargs):

        return self._pennylane_circuit.draw(**kwargs)

    # @abc.abstractmethod
    def set_params(self, **params) -> None:
        raise NotImplementedError

    def get_params(self, deep: bool = True) -> dict:
        """Returns the dictionary of the hyper-parameters of the QNN.

        In case of multiple outputs, the hyper-parameters of the operator are prefixed
        with ``op0__``, ``op1__``, etc.

        """
        params = dict(num_qubits=self.num_qubits)

        if deep:
            params.update(self._pqc.get_params())
            if isinstance(self._observable, list):
                for i, oper in enumerate(self._observable):
                    oper_dict = oper.get_params()
                    for key, value in oper_dict.items():
                        if key != "num_qubits":
                            params["op" + str(i) + "__" + key] = value
            else:
                params.update(self._observable.get_params())
        return params

    @property
    # @abc.abstractmethod
    def num_qubits(self) -> int:
        """Return the number of qubits of the QNN"""
        return self._pennylane_circuit._num_qubits

    @property
    # @abc.abstractmethod
    def num_features(self) -> int:
        """Return the dimension of the features of the PQC"""
        return self._pqc.num_features

    @property
    # @abc.abstractmethod
    def num_parameters(self) -> int:
        """Return the number of trainable parameters of the PQC"""
        return self._pqc.num_parameters

    @property
    def num_operator(self) -> int:
        """Return the number outputs"""
        return self._num_operators

    @property
    def num_parameters_observable(self) -> int:
        """Return the number of trainable parameters of the expectation value operator"""
        return self._num_parameters_observable

    @property
    def multiple_output(self) -> bool:
        """Return true if multiple outputs are used"""
        return self._multiple_output

    def evaluate(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
        *values,  # TODO: data type definition missing Union[str,Expec,tuple,...]
    ) -> dict:

        def _evaluate_todo_single_x(todo_class: direct_evaluation, x, param, param_obs):

            if todo_class.squared:
                func = self._pennylane_circuit_squared
            else:
                func = self._pennylane_circuit

            param_ = pnp.array(param, requires_grad=todo_class.return_grad_param)
            param_obs_ = pnp.array(param_obs, requires_grad=todo_class.return_grad_param_obs)
            x_ = pnp.array(x, requires_grad=todo_class.return_grad_x)

            if todo_class.order == 0:
                value = func(param_, x_, param_obs_)
            elif todo_class.order > 0:
                order = todo_class.order - 1
                argnum = copy.copy(todo_class.argnum)
                deriv = qml.jacobian(func, argnum=argnum.pop())
                while order > 0:
                    deriv = qml.jacobian(deriv, argnum=argnum.pop())
                    order -= 1
                value = deriv(param_, x_, param_obs_)

            return np.array(value)

        def _evaluate_todo_all_x(todo_class: direct_evaluation, x, param, param_obs):

            if todo_class.squared:
                func = self._pennylane_circuit_squared
            else:
                func = self._pennylane_circuit

            param_ = pnp.array(param, requires_grad=todo_class.return_grad_param)
            param_obs_ = pnp.array(param_obs, requires_grad=todo_class.return_grad_param_obs)
            x_ = pnp.array(x, requires_grad=todo_class.return_grad_x)

            if todo_class.order == 0:
                value = func(param_, x_, param_obs_)
            elif todo_class.order > 0:
                order = todo_class.order - 1
                argnum = copy.copy(todo_class.argnum)
                deriv = qml.jacobian(func, argnum=argnum.pop())
                while order > 0:
                    deriv = qml.jacobian(deriv, argnum=argnum.pop())
                    order -= 1
                value = deriv(param_, x_, param_obs_)

            values = np.array(value)
            # sum over zero values entries due to dx differentiation
            if todo_class.return_grad_x:
                sum_t = tuple()
                ioff = 0
                if self.multiple_output:
                    ioff = 1
                for var in reversed(todo_class.argnum):
                    if var == 1:  # dx differentiation
                        sum_t += (ioff + 2,)
                        ioff = ioff + 2
                    else:
                        ioff = ioff + 1
                values = values.sum(axis=sum_t)
            return values

        x_inp, multi_x = adjust_features(x, self._pqc.num_features)
        x_inpT = np.transpose(x_inp)
        param_inp, multi_param = adjust_parameters(param, self._pqc.num_parameters)
        param_obs_inp, multi_param_op = adjust_parameters(
            param_obs, self._num_parameters_observable
        )

        if self._pennylane_circuit.circuit_arguments != ["param", "x", "param_obs"]:
            raise NotImplementedError("Wrong order of circuit arguments!")

        # return dictionary for input data, it will be empty
        # if the combination of x,param,param_op is touched the first time
        if self._result_caching == True:
            caching_tuple = (to_tuple(x), to_tuple(param), to_tuple(param_obs))
            value_dict = self.result_container.get(caching_tuple, {})
        else:
            value_dict = {}

        value_dict["x"] = x
        value_dict["param"] = param
        value_dict["param_op"] = param_obs

        post_processing_values = []

        values = list(values)

        for todo in values:

            todo_class = _get_class_from_string(todo)
            if not isinstance(todo_class,direct_evaluation):
                for sub_todo in todo_class.evaluation_tuple:
                    if sub_todo not in values:
                        values.append(sub_todo)
                post_processing_values.append(todo_class)
            else:

                if todo_class.return_grad_x and todo_class.order > 1:
                    # evaluate every single x, param, param_op combination separately
                    output = [
                        _evaluate_todo_single_x(todo_class, x_inp_, param_inp_, param_obs_inp_)
                        for x_inp_ in x_inp
                        for param_inp_ in param_inp
                        for param_obs_inp_ in param_obs_inp
                    ]
                    output = np.array(output)

                else:
                    # evaluate only param, param_op combination separately and all x together
                    output = [
                        _evaluate_todo_all_x(todo_class, x_inpT, param_inp_, param_obs_inp_)
                        for param_inp_ in param_inp
                        for param_obs_inp_ in param_obs_inp
                    ]
                    # Restore order of _evaluate_todo_single_x
                    output = np.array(output)
                    index_list = list(range(len(output.shape)))
                    if self.multiple_output:
                        swap_list = [2, 0, 1] + index_list[3:]
                    else:
                        swap_list = [1, 0] + index_list[2:]
                    output = output.transpose(swap_list)
                    output = output.reshape(
                        (output.shape[0] * output.shape[1],) + tuple(output.shape[2:])
                    )

                # Swap higher order derivatives into correct order
                index_list = list(range(len(output.shape)))
                if self.multiple_output:
                    swap_list = index_list[0:2] + list(reversed(index_list[2:]))
                else:
                    swap_list = index_list[0:1] + list(reversed(index_list[1:]))
                output = output.transpose(swap_list)

                # Reshape to correct format
                reshape_list = []
                shape = output.shape
                if multi_x:
                    reshape_list.append(len(x))
                if multi_param:
                    reshape_list.append(len(param))
                if multi_param_op:
                    reshape_list.append(len(param_obs))
                if self.multiple_output:
                    reshape_list.append(shape[1])
                if self.multiple_output:
                    reshape_list += list(shape[2:])
                else:
                    reshape_list += list(shape[1:])

                if len(reshape_list) == 0:
                    value_dict[todo] = output.reshape(-1)[0]
                else:
                    value_dict[todo] = output.reshape(reshape_list)


        for post in post_processing_values:
            value_dict[post.key] = post.evaluation_function(value_dict)

        return value_dict

    # #@abc.abstractmethod
    # def evaluate_v2(
    #     self,
    #     x: Union[float, np.ndarray],
    #     param: Union[float, np.ndarray],
    #     param_obs: Union[float, np.ndarray],
    #     *values
    # ) -> dict:

    #     #x,test = adjust_features(x, self._pqc.num_features)

    #     if self._pennylane_circuit.circuit_arguments != ["param","x","param_obs"]:
    #         raise NotImplementedError("Wrong order of circuit arguments!")

    #     value_dict = {}
    #     value_dict["x"] = x
    #     value_dict["param"] = param
    #     value_dict["param_op"] = param_obs
    #     xx = [x] # TODO: multiple features -> in a ([x1][x2]) list format (transposed to out format)

    #     for todo in values:
    #         if todo=="f" or values ==("f",):
    #             param_ = pnp.array(param, requires_grad=False)
    #             param_obs_ = pnp.array(param_obs, requires_grad=False)
    #             x_ = pnp.array(xx, requires_grad=False)
    #             value = np.array(self._pennylane_circuit(param_,x_,param_obs_))
    #             value_dict["f"] = value
    #             # if "f" in value_dict:
    #             #     value_dict["f"].append(value)
    #             # else:
    #             #     value_dict["f"] = [value]
    #         elif todo=="dfdp" or values ==("dfdp",):
    #             param_ = pnp.array(param, requires_grad=True)
    #             param_obs_ = pnp.array(param_obs, requires_grad=False)
    #             x_ = pnp.array(xx, requires_grad=False)
    #             value = np.array(qml.jacobian(self._pennylane_circuit)(param_,x_,param_obs_))
    #             value_dict["dfdp"] = value
    #             # if "dfdp" in value_dict:
    #             #     value_dict["dfdp"].append(value)
    #             # else:
    #             #     value_dict["dfdp"] = [value]
    #         elif todo=="dfdop" or values ==("dfdop",):
    #             param_ = pnp.array(param, requires_grad=False)
    #             param_obs_ = pnp.array(param_obs, requires_grad=True)
    #             x_ = pnp.array(xx, requires_grad=False)
    #             value = np.array(qml.jacobian(self._pennylane_circuit)(param_,x_,param_obs_))
    #             value_dict["dfdop"] = value
    #             # if "dfdop" in value_dict:
    #             #     value_dict["dfdop"].append(value)
    #             # else:
    #             #     value_dict["dfdop"] = [value]
    #         elif todo=="dfdx" or values ==("dfdx",):
    #             param_ = pnp.array(param, requires_grad=False)
    #             param_obs_ = pnp.array(param_obs, requires_grad=False)
    #             x_ = pnp.array(xx, requires_grad=True)
    #             value = np.array(qml.jacobian(self._pennylane_circuit)(param_,x_,param_obs_))
    #             value_dict["dfdx"] = value
    #             # if "dfdx" in value_dict:
    #             #     value_dict["dfdx"].append(value)
    #             # else:
    #             #     value_dict["dfdx"] = [value]

    #     if "f" in value_dict:
    #         value_dict["f"] = np.array(value_dict["f"])
    #     if "dfdp" in value_dict:
    #         value_dict["dfdp"] = np.array(value_dict["dfdp"])
    #     if "dfdop" in value_dict:
    #         value_dict["dfdop"] = np.array(value_dict["dfdop"])
    #     if "dfdx" in value_dict:
    #         value_dict["dfdx"] = np.array(value_dict["dfdx"])

    #     return value_dict

    # def _evaluate_tensorflow(self,values,value_dict,x,param,param_obs):

    #     x = x.transpose()

    #     x_ = tf.convert_to_tensor(x, dtype=tf.float64)
    #     param_ = tf.convert_to_tensor(param, dtype=tf.float64)
    #     param_obs_ = tf.convert_to_tensor(param_obs, dtype=tf.float64)
    #     # TODO -> not 100% working
    #     for todo in values:

    #         if todo in value_dict:
    #             continue

    #         if todo=="f" or values ==("f",):
    #             value = np.array(self._pennylane_circuit(param_,x_,param_obs_))
    #             value_dict["f"] = value
    #         elif todo=="dfdp" or values ==("dfdp",):
    #             value = None
    #             with tf.GradientTape() as tape:
    #                 tape.watch(param_)
    #                 y = self._pennylane_circuit(param_,x_,param_obs_)
    #                 value = tape.gradient(y, param_)
    #             value_dict["dfdp"] = np.array(value)
    #         elif todo=="dfdop" or values ==("dfdop",):
    #             value = None
    #             with tf.GradientTape() as tape:
    #                 tape.watch(param_obs_)
    #                 y = self._pennylane_circuit(param_,x_,param_obs_)
    #                 value = tape.gradient(y, param_obs_)
    #             value_dict["dfdop"] = np.array(value)
    #         elif todo=="dfdx" or values ==("dfdx",):
    #             value = None
    #             with tf.GradientTape() as tape:
    #                 tape.watch(x_)
    #                 y = self._pennylane_circuit(param_,x_,param_obs_)
    #                 value = tape.gradient(y, x_)
    #             value_dict["dfdx"] = np.array(value)

    #     return value_dict

    # def _evaluate_jax(self,values,value_dict,x,param,param_obs):

    #     x = x.transpose()

    #     x_ = jnp.array(x)
    #     param_ = jnp.array(param)
    #     param_obs_ = jnp.array(param_obs)

    #     for todo in values:

    #         if todo in value_dict:
    #             continue

    #         if todo=="f" or values ==("f",):
    #             value = np.array(self._pennylane_circuit(param_,x_,param_obs_))
    #             value_dict["f"] = value
    #         elif todo=="dfdp" or values ==("dfdp",):

    #             if "dfdp" not in self._jax_cache:
    #                 self._jax_cache["dfdp"] = jax.jacobian(self._pennylane_circuit, argnums=0)
    #             fun = self._jax_cache["dfdp"]
    #             value = fun(param_,x_,param_obs_)
    #             value_dict["dfdp"] = np.array(value)
    #         elif todo=="dfdop" or values ==("dfdop",):

    #             if "dfdop" not in self._jax_cache:
    #                 self._jax_cache["dfdop"] = jax.jacobian(self._pennylane_circuit, argnums=2)
    #             fun = self._jax_cache["dfdop"]
    #             value = fun(param_,x_,param_obs_)
    #             value_dict["dfdop"] = np.array(value)
    #         elif todo=="dfdx" or values ==("dfdx",):

    #             if "dfdx" not in self._jax_cache:
    #                 self._jax_cache["dfdx"] = jax.jacobian(self._pennylane_circuit, argnums=1)
    #             fun = self._jax_cache["dfdx"]
    #             value = fun(param_,x_,param_obs_)
    #             value_dict["dfdx"] = np.array(value)

    #     return value_dict

    # def _evaluate_pytorch(self,values,value_dict,x,param,param_obs):

    #     x = x.transpose()
    #     print(x)

    #     for todo in values:

    #         if todo in value_dict:
    #             continue

    #         if todo=="f" or values ==("f",):
    #             x_ = torch.tensor(x, dtype=torch.float64, requires_grad=False)
    #             param_ = torch.tensor(param, dtype=torch.float64, requires_grad=False)
    #             param_obs_ = torch.tensor(param_obs, dtype=torch.float64, requires_grad=False)
    #             value = np.array(self._pennylane_circuit(param_,x_,param_obs_))
    #             value_dict["f"] = value
    #         elif todo=="dfdp" or values ==("dfdp",):
    #             x_ = torch.tensor(x, requires_grad=False)
    #             param_ = torch.tensor(param, requires_grad=True)
    #             param_obs_ = torch.tensor(param_obs, requires_grad=False)
    #             result = self._pennylane_circuit(param_,x_,param_obs_)
    #             print("x_",x_)
    #             print("result",result)
    #             print("torch.ones_like(result)",torch.ones_like(result))
    #             value = torch.autograd.grad(result, param_, torch.ones_like(result),retain_graph=True )#, create_graph =True)
    #             print("value",value)
    #             value_dict["dfdp"] = np.array(value)
    #         elif todo=="dfdop" or values ==("dfdop",):
    #             x_ = torch.tensor(x, dtype=torch.float64, requires_grad=False)
    #             param_ = torch.tensor(param, dtype=torch.float64, requires_grad=False)
    #             param_obs_ = torch.tensor(param_obs, dtype=torch.float64, requires_grad=True)
    #             result = self._pennylane_circuit(param_,x_,param_obs_)
    #             result.backward(torch.ones_like(result),create_graph=True)
    #             value_dict["dfdop"] = np.array(param_obs_.grad)
    #         elif todo=="dfdx" or values ==("dfdx",):
    #             x_ = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    #             param_ = torch.tensor(param, dtype=torch.float64, requires_grad=False)
    #             param_obs_ = torch.tensor(param_obs, dtype=torch.float64, requires_grad=False)
    #             result = self._pennylane_circuit(param_,x_,param_obs_)
    #             result.backward(torch.ones_like(result),create_graph=True)
    #             value_dict["dfdx"] = np.array(x_.grad)

    #     return value_dict

    def evaluate_f(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
    ) -> dict:

        return self.evaluate(x, param, param_obs, "f")["f"]
