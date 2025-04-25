"""Tests for FidelityKernel"""

import pytest
import numpy as np
import sympy as sp

from unittest.mock import MagicMock

from squlearn import Executor
from squlearn.encoding_circuit import LayeredEncodingCircuit
from squlearn.kernel.lowlevel_kernel import FidelityKernel


class TestFidelityKernel:
    """Test class for FidelityKernels"""

    @pytest.fixture
    def setup_single_variable(self):
        x, y, p = sp.symbols("x y p")
        sympy_K = (
            0.375 * (1 - sp.cos(p)) ** 2 * sp.cos(x - y)
            + 0.125 * (1 - sp.cos(p)) ** 2 * sp.cos(x + y)
            - 0.5 * (1 - sp.cos(p)) ** 2
            - 1.0 * sp.cos(p)
            - 0.3125 * sp.cos(x - y)
            - 0.1875 * sp.cos(x + y)
            - 0.03125 * sp.cos(-2 * p + x + y)
            + 0.125 * sp.cos(-p + x + y)
            + 0.375 * sp.cos(p - x + y)
            + 0.375 * sp.cos(p + x - y)
            + 0.125 * sp.cos(p + x + y)
            + 0.03125 * sp.cos(2 * p - x + y)
            + 0.03125 * sp.cos(2 * p + x - y)
            - 0.03125 * sp.cos(2 * p + x + y)
            + 1.5
        )

        sympy_values = {
            "K": sympy_K,
            "dKdx": sp.diff(sympy_K, x),
            "dKdy": sp.diff(sympy_K, y),
            "dKdxdx": sp.diff(sp.diff(sympy_K, x), x),
            "dKdp": sp.diff(sympy_K, p),
        }

        return x, y, p, sympy_values

    @pytest.fixture
    def setup_multi_variable(self):
        x0, y0, p0 = sp.symbols("x0 y0 p0")
        x1, y1, p1 = sp.symbols("x1 y1 p1")

        sympy_K = (
            0.25 * sp.cos(x0 - y0)
            + 0.25 * sp.cos(x1 - y1)
            + 0.125 * sp.cos(x0 - x1 - y0 + y1)
            + 0.125 * sp.cos(x0 + x1 - y0 - y1)
            + 0.25
        )

        sympy_values = {
            "K": sympy_K,
            "dKdx0": sp.diff(sympy_K, x0),
            "dKdx1": sp.diff(sympy_K, x1),
            "dKdy0": sp.diff(sympy_K, y0),
            "dKdy1": sp.diff(sympy_K, y1),
            "dKdp0": sp.diff(sympy_K, p0),
            "dKdp1": sp.diff(sympy_K, p1),
        }

        return x0, y0, x1, y1, p0, p1, sympy_values

    def create_fidelity_kernel(
        self, num_features, initial_parameters, executor, use_expectation, evaluate_duplicates
    ):
        executor = executor
        encoding_circuit = LayeredEncodingCircuit(
            num_qubits=num_features, num_features=num_features
        )

        if num_features == 1:
            encoding_circuit.Ry("p")  # For 1 feature, the FQK analytical kernel is:  Rx(x)@Ry(p)
        elif num_features == 2:
            encoding_circuit.Rx(
                "p"
            )  # For 2 features, the FQK analytical kernel is:  Rx(x0)@Ry(p0) x Rx(x1)@Ry(p1)
        encoding_circuit.Rx("x")

        kernel = FidelityKernel(
            encoding_circuit,
            executor=executor,
            caching=False,
            use_expectation=use_expectation,
            initial_parameters=initial_parameters,
            evaluate_duplicates=evaluate_duplicates,
        )

        return kernel

    def test_single_variable_derivatives(self, setup_single_variable):
        x, y, p, sympy_values = setup_single_variable

        x_num, y_num = 0.79, -0.31
        p_num = -0.63

        subs = {x: x_num, y: y_num, p: p_num}
        sympy_num_values = {key: sympy_values[key].evalf(subs=subs) for key in sympy_values}

        kernel = self.create_fidelity_kernel(1, [p_num], Executor("pennylane"), True, "all")

        values = kernel.evaluate_derivatives(
            np.array([[x_num]]), np.array([[y_num]]), ["K", "dKdx", "dKdy", "dKdxdx", "dKdp"]
        )
        for key in ["K", "dKdx", "dKdy", "dKdxdx", "dKdp"]:
            assert np.allclose(
                np.array(values[key]).flatten().astype(float),
                np.array(sympy_num_values[key]).astype(float),
                atol=1e-7,
            )

    def test_multi_variable_derivatives(self, setup_multi_variable):
        x0, y0, x1, y1, p0, p1, sympy_values = setup_multi_variable

        x0_num, y0_num = 0.79, -0.31
        x1_num, y1_num = 0.9, -1.31
        p_num = -0.63

        subs = {
            x0: x0_num,
            y0: y0_num,
            x1: x1_num,
            y1: y1_num,
            p0: p_num,
            p1: p_num,
        }
        sympy_num_values = {
            "K": sympy_values["K"].evalf(subs=subs),
            "dKdx": [
                sympy_values["dKdx0"].evalf(subs=subs),
                sympy_values["dKdx1"].evalf(subs=subs),
            ],
            "dKdy": [
                sympy_values["dKdy0"].evalf(subs=subs),
                sympy_values["dKdy1"].evalf(subs=subs),
            ],
            "dKdp": [
                sympy_values["dKdp0"].evalf(subs=subs),
                sympy_values["dKdp1"].evalf(subs=subs),
            ],
        }

        kernel = self.create_fidelity_kernel(2, [p_num, p_num], Executor("pennylane"), True, "all")

        values = kernel.evaluate_derivatives(
            np.array([[x0_num, x1_num]]),
            np.array([[y0_num, y1_num]]),
            ["K", "dKdx", "dKdy", "dKdp"],
        )

        for key in ["K", "dKdx", "dKdy", "dKdp"]:
            assert np.allclose(
                np.array(values[key]).flatten().astype(float),
                np.array(sympy_num_values[key]).astype(float),
                atol=1e-7,
            )

    def test_with_and_without_duplicates(self):
        np.random.seed(0)
        x_train = np.random.rand(10, 2)
        p_num = -0.63

        # Check if FidelityKernelExpectationValue and FidelityKernelPennylane give the same results for no duplicates
        kernel_without_duplicates_and_expectation_value = self.create_fidelity_kernel(
            2, [p_num, p_num], Executor("pennylane"), True, "none"
        )

        kernel_without_duplicates_and_pure_pennylane = self.create_fidelity_kernel(
            2, [p_num, p_num], Executor("pennylane"), False, "none"
        )

        values_expecation_value = (
            kernel_without_duplicates_and_expectation_value.evaluate_derivatives(
                x_train, x_train, "K"
            )
        )
        values_pennylane = kernel_without_duplicates_and_pure_pennylane.evaluate(x_train, x_train)
        assert np.allclose(values_expecation_value, values_pennylane, atol=1e-7)
