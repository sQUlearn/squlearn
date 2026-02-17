import pytest
import numpy as np
from packaging import version

from qiskit import __version__ as qiskit_version
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector

from squlearn.util import OpTree

QISKIT_SMALLER_1_0 = version.parse(qiskit_version) < version.parse("1.0.0")


class TestOpTreeDerivative:
    """Test class for OpTree derivatives."""

    def test_derivative(self):
        """Function for comparing analytical and numerical derivatives"""

        p = ParameterVector("p", 1)

        qc = QuantumCircuit(2)
        qc.rx(2.0 * p[0], 0)
        qc.ry(3.0 * p[0] + 0.5, 1)
        qc.cx(0, 1)

        operator = SparsePauliOp(["IZ", "ZI"])

        p_val = np.linspace(-0.49, 0.49, 100)
        p_array = [{p[0]: p_} for p_ in p_val]

        if QISKIT_SMALLER_1_0:
            from qiskit.primitives import Estimator

            estimator = Estimator()
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator(default_precision=0.0)

        val = OpTree.evaluate.evaluate_with_estimator(qc, operator, p_array, {}, estimator)
        qc_d = OpTree.derivative.differentiate(qc, p[0])
        val_d = OpTree.evaluate.evaluate_with_estimator(qc_d, operator, p_array, {}, estimator)
        qc_dd = OpTree.derivative.differentiate(qc_d, p[0])
        val_dd = OpTree.evaluate.evaluate_with_estimator(qc_dd, operator, p_array, {}, estimator)

        val_d_num = np.gradient(val, p_val)
        val_dd_num = np.gradient(val_d_num[1:-1], p_val[1:-1])

        # Compare numerical and analytical derivatives
        assert np.allclose(val_d_num[1:-1], val_d[1:-1], rtol=1e-2)
        # Second derivative has slightly larger tolerance (4e-2) due to numerical errors
        # that accumulate when taking gradients of gradients
        assert np.allclose(val_dd_num[1:-1], val_dd[2:-2], rtol=4e-2)

    def test_qc_gradient(self):
        """Function for testing derivatives of the circuit"""

        # set-up of the expectation value
        p = ParameterVector("p", 4)
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(2)
        # Use linear encoding of x (not arccos) to allow differentiation w.r.t. x
        qc.rx(p[0] * x[0], 0)
        qc.rx(p[1] * x[0], 1)
        qc.ry(p[2], 0)
        qc.ry(p[3], 1)
        qc.rxx(p[0] * x[0], 0, 1)
        operator = SparsePauliOp(["IZ", "ZI"])
        dictionary = {x[0]: 0.5, p[0]: 1.5, p[1]: 2.5, p[2]: 0.5, p[3]: 0.25}

        if QISKIT_SMALLER_1_0:
            from qiskit.primitives import Estimator

            estimator = Estimator()
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator(default_precision=0.0)

        # Compare the gradient w.r.t the parameters p to precomputed values
        qc_grad = OpTree.derivative.differentiate(qc, p)
        qc_grad_v2 = OpTree.derivative.differentiate_v2(qc, p)
        # Note: Reference values updated for linear encoding instead of arccos encoding
        reference_grad = np.array([-0.59681901, -0.31954279, -0.67203245, -0.19903458])
        assert np.allclose(
            OpTree.evaluate.evaluate_with_estimator(qc_grad, operator, dictionary, {}, estimator),
            reference_grad,
        )
        assert np.allclose(
            OpTree.evaluate.evaluate_with_estimator(
                qc_grad_v2, operator, dictionary, {}, estimator
            ),
            reference_grad,
        )

        # Compare the gradient w.r.t x to precomputed values
        qc_dx = OpTree.derivative.differentiate(qc, x)
        qc_dx_v2 = OpTree.derivative.differentiate_v2(qc, x)
        # Note: Reference values updated for linear encoding instead of arccos encoding
        reference_dx = np.array([-3.38817095])

        assert np.allclose(
            OpTree.evaluate.evaluate_with_estimator(qc_dx, operator, dictionary, {}, estimator),
            reference_dx,
        )
        assert np.allclose(
            OpTree.evaluate.evaluate_with_estimator(qc_dx_v2, operator, dictionary, {}, estimator),
            reference_dx,
        )

    def test_operator_gradient(self):
        """Function for testing derivatives of the operator"""

        p = ParameterVector("p", 4)
        dictionary_p = {p[0]: 1.5, p[1]: 2.5, p[2]: 0.5, p[3]: 0.25}

        operator = SparsePauliOp(["IZ", "ZI", "IX", "XI"], [p[0], p[1], p[2], p[3]])
        operator = operator.power(2)  # square operator for a more complicated operator
        # trivial circuit
        qc = QuantumCircuit(2)
        qc.h([0, 1])

        if QISKIT_SMALLER_1_0:
            from qiskit.primitives import Estimator

            estimator = Estimator()
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator(default_precision=0.0)

        # Check if the gradient reproduces the correct values
        op_grad = OpTree.derivative.differentiate(operator, p)
        op_grad_v2 = OpTree.derivative.differentiate_v2(operator, p)
        reference_values = np.array([3.0, 5.0, 1.5, 1.5])
        assert np.allclose(
            OpTree.evaluate.evaluate_with_estimator(qc, op_grad, {}, dictionary_p, estimator),
            reference_values,
        )
        assert np.allclose(
            OpTree.evaluate.evaluate_with_estimator(qc, op_grad_v2, {}, dictionary_p, estimator),
            reference_values,
        )

        if QISKIT_SMALLER_1_0:
            from qiskit.primitives import Sampler

            sampler = Sampler()
        else:
            from qiskit.primitives import StatevectorSampler

            sampler = StatevectorSampler(seed=0, default_shots=10000)
            reference_values = np.array(
                [
                    3.028,
                    5.0154,
                    1.49,
                    1.494,
                ]
            )

        # Check if gradient works with a derivative of the z-basis transformed operator
        operator_z = OpTree.evaluate.transform_to_zbasis(operator)
        op_grad_z = OpTree.derivative.differentiate(operator_z, p)
        op_grad_z_v2 = OpTree.derivative.differentiate_v2(operator_z, p)
        assert np.allclose(
            OpTree.evaluate.evaluate_with_sampler(qc, op_grad_z, {}, dictionary_p, sampler),
            reference_values,
        )
        assert np.allclose(
            OpTree.evaluate.evaluate_with_sampler(qc, op_grad_z_v2, {}, dictionary_p, sampler),
            reference_values,
        )

        # Check if gradient works with a z-basis transformed operator
        assert np.allclose(
            OpTree.evaluate.evaluate_with_sampler(
                qc, OpTree.evaluate.transform_to_zbasis(op_grad), {}, dictionary_p, sampler
            ),
            reference_values,
        )
        assert np.allclose(
            OpTree.evaluate.evaluate_with_sampler(
                qc, OpTree.evaluate.transform_to_zbasis(op_grad_v2), {}, dictionary_p, sampler
            ),
            reference_values,
        )

    def test_nonlinear_parameter_error(self):
        """Test that non-linear parameters raise an error"""

        p = ParameterVector("p", 1)

        # Test with arccos (non-linear function)
        qc_arccos = QuantumCircuit(1)
        qc_arccos.rx(np.arccos(p[0]), 0)

        with pytest.raises(ValueError, match="Parameter shift rule cannot be applied"):
            OpTree.derivative.differentiate(qc_arccos, p[0])

        # Test with sin (non-linear function)
        qc_sin = QuantumCircuit(1)
        qc_sin.rx(np.sin(p[0]), 0)

        with pytest.raises(ValueError, match="Parameter shift rule cannot be applied"):
            OpTree.derivative.differentiate(qc_sin, p[0])

        # Test with quadratic (non-linear)
        qc_quad = QuantumCircuit(1)
        qc_quad.rx(p[0] ** 2, 0)

        with pytest.raises(ValueError, match="Parameter shift rule cannot be applied"):
            OpTree.derivative.differentiate(qc_quad, p[0])

        # Test that linear parameters still work
        qc_linear = QuantumCircuit(1)
        qc_linear.rx(2.0 * p[0] + 1.0, 0)

        # This should not raise an error
        qc_linear_d = OpTree.derivative.differentiate(qc_linear, p[0])
        assert qc_linear_d is not None
