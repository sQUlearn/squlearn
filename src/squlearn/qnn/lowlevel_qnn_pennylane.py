import abc
from typing import Union
import numpy as np

from qiskit.circuit import ParameterVector

import pennylane as qml
import pennylane.numpy as pnp

from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ..util import Executor

from ..util.pennylane import PennyLaneCircuit, PennyLaneDevice

from .lowlevel_qnn_base import LowLevelQNNBase

from ..util.data_preprocessing import adjust_features, to_tuple

import tensorflow as tf

import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

import torch

class LowLevelQNNPennyLane(LowLevelQNNBase):

    def __init__(self,
                 pqc: EncodingCircuitBase,
                 observable: Union[ObservableBase, list],
                 executor: Executor,
                 result_caching: bool = False,
                 ) -> None:

        super().__init__(pqc, observable, executor)

        self._device = PennyLaneDevice(gradient_engine="pytorch")

        self._x = ParameterVector("x", self._pqc.num_features)
        self._param = ParameterVector("param", self._pqc.num_parameters)
        self._param_obs = ParameterVector("param_obs", self._observable.num_parameters)
        self._qiskit_circuit = self._pqc.get_circuit(self._x, self._param)
        self._qiskit_observable = self._observable.get_operator(self._param_obs)

        self._pennylane_circuit = PennyLaneCircuit(self._device, self._qiskit_circuit, self._qiskit_observable)

        # Initialize result cache
        self._result_caching = result_caching
        self.result_container = {}

        self._jax_cache = {}

    def draw(self,**kwargs):

        return self._pennylane_circuit.draw(**kwargs)


    #@abc.abstractmethod
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
    #@abc.abstractmethod
    def num_qubits(self) -> int:
        """Return the number of qubits of the QNN"""
        return self._pennylane_circuit._num_qubits

    @property
    #@abc.abstractmethod
    def num_features(self) -> int:
        """Return the dimension of the features of the PQC"""
        return self._pqc.num_features

    @property
    #@abc.abstractmethod
    def num_parameters(self) -> int:
        """Return the number of trainable parameters of the PQC"""
        return self._pqc.num_parameters

    @property
    def num_operator(self) -> int:
        """Return the number outputs"""
        return 1 #TODO

    @property
    def num_parameters_observable(self) -> int:
        """Return the number of trainable parameters of the expectation value operator"""
        return self._observable.num_parameters

    @property
    def multiple_output(self) -> bool:
        """Return true if multiple outputs are used"""
        return False # TODO

    #@abc.abstractmethod
    def evaluate(
        self,
        values,  # TODO: data type definition missing Union[str,Expec,tuple,...]
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
    ) -> dict:

        xx,test = adjust_features(x, self._pqc.num_features)

        if not isinstance(values, tuple):
            values = (values,)

        if self._pennylane_circuit.circuit_arguments != ["param","x","param_obs"]:
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

        if self._device._gradient_engine == "autodiff":
            value_dict = self._evaluate_autograd(values,value_dict,xx,param,param_obs)
        elif self._device._gradient_engine == "tf" or self._device._gradient_engine == "tensorflow":
            value_dict = self._evaluate_tensorflow(values,value_dict,xx,param,param_obs)
        elif self._device._gradient_engine == "jax":
            value_dict = self._evaluate_jax(values,value_dict,xx,param,param_obs)
        elif self._device._gradient_engine == "torch" or self._device._gradient_engine == "pytorch":
            value_dict = self._evaluate_pytorch(values,value_dict,xx,param,param_obs)
        else:
            raise NotImplementedError("Gradient engine not implemented")


        return value_dict



    def _evaluate_autograd(self,values,value_dict,x,param,param_obs):

        x = x.transpose()

        for todo in values:

            if todo in value_dict:
                continue

            if todo=="f" or values ==("f",):
                param_ = pnp.array(param, requires_grad=False)
                param_obs_ = pnp.array(param_obs, requires_grad=False)
                x_ = pnp.array(x, requires_grad=False)
                value = np.array(self._pennylane_circuit(param_,x_,param_obs_))
                value_dict["f"] = value
            elif todo=="dfdp" or values ==("dfdp",):
                param_ = pnp.array(param, requires_grad=True)
                param_obs_ = pnp.array(param_obs, requires_grad=False)
                x_ = pnp.array(x, requires_grad=False)
                value = np.array(qml.jacobian(self._pennylane_circuit)(param_,x_,param_obs_))
                value_dict["dfdp"] = value
            elif todo=="dfdop" or values ==("dfdop",):
                param_ = pnp.array(param, requires_grad=False)
                param_obs_ = pnp.array(param_obs, requires_grad=True)
                x_ = pnp.array(x, requires_grad=False)
                value = np.array(qml.jacobian(self._pennylane_circuit)(param_,x_,param_obs_))
                value_dict["dfdop"] = value
            elif todo=="dfdx" or values ==("dfdx",):
                param_ = pnp.array(param, requires_grad=False)
                param_obs_ = pnp.array(param_obs, requires_grad=False)
                x_ = pnp.array(x, requires_grad=True)
                value = np.array(qml.jacobian(self._pennylane_circuit)(param_,x_,param_obs_))
                value_dict["dfdx"] = value

        return value_dict

    def _evaluate_tensorflow(self,values,value_dict,x,param,param_obs):

        x = x.transpose()

        x_ = tf.convert_to_tensor(x, dtype=tf.float64)
        param_ = tf.convert_to_tensor(param, dtype=tf.float64)
        param_obs_ = tf.convert_to_tensor(param_obs, dtype=tf.float64)
        # TODO -> not 100% working
        for todo in values:

            if todo in value_dict:
                continue

            if todo=="f" or values ==("f",):
                value = np.array(self._pennylane_circuit(param_,x_,param_obs_))
                value_dict["f"] = value
            elif todo=="dfdp" or values ==("dfdp",):
                value = None
                with tf.GradientTape() as tape:
                    tape.watch(param_)
                    y = self._pennylane_circuit(param_,x_,param_obs_)
                    value = tape.gradient(y, param_)
                value_dict["dfdp"] = np.array(value)
            elif todo=="dfdop" or values ==("dfdop",):
                value = None
                with tf.GradientTape() as tape:
                    tape.watch(param_obs_)
                    y = self._pennylane_circuit(param_,x_,param_obs_)
                    value = tape.gradient(y, param_obs_)
                value_dict["dfdop"] = np.array(value)
            elif todo=="dfdx" or values ==("dfdx",):
                value = None
                with tf.GradientTape() as tape:
                    tape.watch(x_)
                    y = self._pennylane_circuit(param_,x_,param_obs_)
                    value = tape.gradient(y, x_)
                value_dict["dfdx"] = np.array(value)

        return value_dict

    def _evaluate_jax(self,values,value_dict,x,param,param_obs):

        x = x.transpose()

        x_ = jnp.array(x)
        param_ = jnp.array(param)
        param_obs_ = jnp.array(param_obs)

        for todo in values:

            if todo in value_dict:
                continue

            if todo=="f" or values ==("f",):
                value = np.array(self._pennylane_circuit(param_,x_,param_obs_))
                value_dict["f"] = value
            elif todo=="dfdp" or values ==("dfdp",):

                if "dfdp" not in self._jax_cache:
                    self._jax_cache["dfdp"] = jax.jacobian(self._pennylane_circuit, argnums=0)
                fun = self._jax_cache["dfdp"]
                value = fun(param_,x_,param_obs_)
                value_dict["dfdp"] = np.array(value)
            elif todo=="dfdop" or values ==("dfdop",):

                if "dfdop" not in self._jax_cache:
                    self._jax_cache["dfdop"] = jax.jacobian(self._pennylane_circuit, argnums=2)
                fun = self._jax_cache["dfdop"]
                value = fun(param_,x_,param_obs_)
                value_dict["dfdop"] = np.array(value)
            elif todo=="dfdx" or values ==("dfdx",):

                if "dfdx" not in self._jax_cache:
                    self._jax_cache["dfdx"] = jax.jacobian(self._pennylane_circuit, argnums=1)
                fun = self._jax_cache["dfdx"]
                value = fun(param_,x_,param_obs_)
                value_dict["dfdx"] = np.array(value)

        return value_dict

    def _evaluate_pytorch(self,values,value_dict,x,param,param_obs):

        x = x.transpose()
        print(x)

        for todo in values:

            if todo in value_dict:
                continue

            if todo=="f" or values ==("f",):
                x_ = torch.tensor(x, dtype=torch.float64, requires_grad=False)
                param_ = torch.tensor(param, dtype=torch.float64, requires_grad=False)
                param_obs_ = torch.tensor(param_obs, dtype=torch.float64, requires_grad=False)
                value = np.array(self._pennylane_circuit(param_,x_,param_obs_))
                value_dict["f"] = value
            elif todo=="dfdp" or values ==("dfdp",):
                x_ = torch.tensor(x, requires_grad=False)
                param_ = torch.tensor(param, requires_grad=True)
                param_obs_ = torch.tensor(param_obs, requires_grad=False)
                result = self._pennylane_circuit(param_,x_,param_obs_)
                print("x_",x_)
                print("result",result)
                print("torch.ones_like(result)",torch.ones_like(result))
                value = torch.autograd.grad(result, param_, torch.ones_like(result),retain_graph=True )#, create_graph =True)
                print("value",value)
                value_dict["dfdp"] = np.array(value)
            elif todo=="dfdop" or values ==("dfdop",):
                x_ = torch.tensor(x, dtype=torch.float64, requires_grad=False)
                param_ = torch.tensor(param, dtype=torch.float64, requires_grad=False)
                param_obs_ = torch.tensor(param_obs, dtype=torch.float64, requires_grad=True)
                result = self._pennylane_circuit(param_,x_,param_obs_)
                result.backward(torch.ones_like(result),create_graph=True)
                value_dict["dfdop"] = np.array(param_obs_.grad)
            elif todo=="dfdx" or values ==("dfdx",):
                x_ = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                param_ = torch.tensor(param, dtype=torch.float64, requires_grad=False)
                param_obs_ = torch.tensor(param_obs, dtype=torch.float64, requires_grad=False)
                result = self._pennylane_circuit(param_,x_,param_obs_)
                result.backward(torch.ones_like(result),create_graph=True)
                value_dict["dfdx"] = np.array(x_.grad)

        return value_dict







    def evaluate_f(self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
    ) -> dict:

        return self.evaluate("f",x, param, param_obs)["f"]