"""Tests for ProjectedQuantumKernel"""

import pytest
import numpy as np
import sympy as sp

from unittest.mock import MagicMock

from squlearn import Executor
from squlearn.encoding_circuit import LayeredEncodingCircuit
from squlearn.kernel.lowlevel_kernel import ProjectedQuantumKernel


class TestProjectedQuantumKernel:
    """Test class for ProjectedQuantumKernel"""

    @pytest.fixture
    def setup_single_variable(self):
        x, y, gamma_sp, p = sp.symbols("x y gamma p")
        sympy_K = sp.exp(-2 * gamma_sp * (1 - sp.cos(x - y)) * sp.cos(p) ** 2)

        sympy_values = {
            "K": sympy_K,
            "dKdx": sp.diff(sympy_K, x),
            "dKdy": sp.diff(sympy_K, y),
            "dKdxdx": sp.diff(sp.diff(sympy_K, x), x),
            "dKdp": sp.diff(sympy_K, p),
        }

        return x, y, gamma_sp, p, sympy_values

    @pytest.fixture
    def setup_multi_variable(self):
        x0, y0, gamma_sp, p0 = sp.symbols("x0 y0 gamma p0")
        x1, y1, p1 = sp.symbols("x1 y1 p1")

        sympy_K = sp.exp(
            -2.0
            * gamma_sp
            * (
                -sp.cos(p0) ** 2 * sp.cos(x0 - y0)
                + sp.cos(p0) ** 2
                - sp.cos(p1) ** 2 * sp.cos(x1 - y1)
                + sp.cos(p1) ** 2
            )
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

        return x0, y0, x1, y1, gamma_sp, p0, p1, sympy_values

    def create_projected_quantum_kernel(self, num_features, gamma_num, initial_parameters):
        executor = Executor()
        encoding_circuit = LayeredEncodingCircuit(
            num_qubits=num_features, num_features=num_features
        )
        encoding_circuit.Ry("p")
        encoding_circuit.Rx("x")

        kernel = ProjectedQuantumKernel(
            encoding_circuit=encoding_circuit,
            executor=executor,
            outer_kernel="gaussian",
            gamma=gamma_num,
            initial_parameters=np.array(initial_parameters),
        )
        return kernel

    def test_single_variable_derivatives(self, setup_single_variable):
        x, y, gamma_sp, p, sympy_values = setup_single_variable

        x_num, y_num = 0.79, -0.31
        p_num = -0.63
        gamma_num = 0.08

        subs = {x: x_num, y: y_num, gamma_sp: gamma_num, p: p_num}
        sympy_num_values = {key: sympy_values[key].evalf(subs=subs) for key in sympy_values}

        kernel = self.create_projected_quantum_kernel(1, gamma_num, [p_num])

        values = kernel.evaluate_derivatives(
            [x_num], [y_num], ["K", "dKdx", "dKdy", "dKdxdx", "dKdp"]
        )
        for key in ["K", "dKdx", "dKdy", "dKdxdx", "dKdp"]:
            assert np.allclose(
                np.array(values[key]).flatten().astype(float),
                np.array(sympy_num_values[key]).astype(float),
                atol=1e-7,
            )

    def test_multi_variable_derivatives(self, setup_multi_variable):
        x0, y0, x1, y1, gamma_sp, p0, p1, sympy_values = setup_multi_variable

        x0_num, y0_num = 0.79, -0.31
        x1_num, y1_num = 0.9, -1.31
        p_num = -0.63
        gamma_num = 0.08

        subs = {
            x0: x0_num,
            y0: y0_num,
            x1: x1_num,
            y1: y1_num,
            gamma_sp: gamma_num,
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

        kernel = self.create_projected_quantum_kernel(2, gamma_num, [p_num, p_num])

        values = kernel.evaluate_derivatives(
            [[x0_num, x1_num]], [[y0_num, y1_num]], ["K", "dKdx", "dKdy", "dKdp"]
        )

        for key in ["K", "dKdx", "dKdy", "dKdp"]:
            assert np.allclose(
                np.array(values[key]).flatten().astype(float),
                np.array(sympy_num_values[key]).astype(float),
                atol=1e-7,
            )
