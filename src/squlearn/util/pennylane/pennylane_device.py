from typing import Union

import pennylane as qml
import pennylane.numpy as pnp

# TODO: Implement dynamic import for tensorflow, jax and torch


import tensorflow as tf

import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

import torch



class PennyLaneDevice():

    def __init__(self, name: str = "default.qubit", num_qubits: Union[int,None]=None, gradient_engine: str="autodiff"):

        self._device_name = name
        self._wires = num_qubits

        if self._wires is not None:
            self._device = qml.device(
                    self._device_name, wires=self._wires
                )
        else:
            self._device = qml.device(
                    self._device_name
                )

        self._gradient_engine = "autodiff" # autodiff, tf, torch, jax

    @property
    def device(self) -> qml.device:
        return self._device

    @property
    def wires(self) -> int:
        return self._wires

    @property
    def num_qubits(self) -> int:
        return self.wires

    @property
    def gradient_engine(self) -> str:
        return self._gradient_engine

    def set_num_qubits(self, num_qubits):

        if self._wires != num_qubits:
            self._wires = num_qubits
            self._device = qml.device(
                    self._device_name, wires=self._wires
                )

    def add_pennylane_decorator(self, pennylane_function):

            if self._gradient_engine == "autodiff":
                return qml.qnode(self._device, diff_method="backprop", interface="autograd")(pennylane_function)
            elif self._gradient_engine == "tf" or self._gradient_engine == "tensorflow":
                return qml.qnode(self._device, diff_method="backprop", interface="tf")(pennylane_function)
            elif self._gradient_engine == "jax":
                return qml.qnode(self._device, diff_method="backprop", interface="jax")(pennylane_function)
            elif self._gradient_engine == "torch" or self._gradient_engine == "pytorch":
                return qml.qnode(self._device, diff_method="backprop", interface="torch")(pennylane_function)
            else:
                raise NotImplementedError("Gradient engine not implemented")

    def get_sympy_interface(self):

        if self._gradient_engine == "autodiff":
            # SymPy printer for pennylane numpy implementation has to be set manually,
            # otherwise math functions are used in lambdify instead of pennylane.numpy functions
            from sympy.printing.numpy import NumPyPrinter as Printer

            user_functions = {}
            printer = Printer(
                {
                    "fully_qualified_modules": False,
                    "inline": True,
                    "allow_unknown_functions": True,
                    "user_functions": user_functions,
                }
            )  #
            modules = pnp
        elif self._gradient_engine == "tf" or self._gradient_engine == "tensorflow":

            # SymPy printer for pennylane numpy implementation has to be set manually,
            # otherwise math functions are used in lambdify instead of pennylane.numpy functions
            from sympy.printing.tensorflow import TensorflowPrinter as Printer # type: ignore

            user_functions = {}
            printer = Printer(
                {
                    "fully_qualified_modules": False,
                    "inline": True,
                    "allow_unknown_functions": True,
                    "user_functions": user_functions,
                }
            )  #
            modules = tf

        elif self._gradient_engine == "jax":
            from sympy.printing.numpy import JaxPrinter as Printer # type: ignore
            user_functions = {}
            printer = Printer(
                {
                    "fully_qualified_modules": False,
                    "inline": True,
                    "allow_unknown_functions": True,
                    "user_functions": user_functions,
                }
            )  #
            modules = jnp
        elif self._gradient_engine == "torch" or self._gradient_engine == "pytorch":
            from sympy.printing.pycode import PythonCodePrinter as Printer # type: ignore

            user_functions = {}
            printer = Printer(
                {
                    "fully_qualified_modules": False,
                    "inline": True,
                    "allow_unknown_functions": True,
                    "user_functions": user_functions,
                }
            )  #
            modules = torch

        else:
            # tbd for jax and tensorflow
            printer = None
            modules = None

        return printer, modules
