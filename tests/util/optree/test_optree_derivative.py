import pytest
from typing import Tuple, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator, Sampler

from squlearn.util import OpTree
from squlearn.util.optree import OpTreeSum, OpTreeList



class TestOpTreeDerivative:
    """Test class for OpTree evaluation"""

    def test_derivative(self):

        p = ParameterVector('p', 1)

        qc = QuantumCircuit(2)
        qc.rx(2.0*p[0], 0)
        qc.rx(10.0*np.arccos(p[0]), 1)
        qc.cx(0, 1)

        operator = SparsePauliOp(["IZ", "ZI"])

        p_val = np.arange(-0.5, 0.5, 0.01)
        p_array = [{p[0]:p_} for p_ in p_val]
        val = OpTree.evaluate.evaluate_with_estimator(qc, operator, p_array, {},Estimator())
        qc_d = OpTree.derivative.differentiate(qc, p[0])
        val_d = OpTree.evaluate.evaluate_with_estimator(qc_d, operator, p_array, {},Estimator())
        qc_dd = OpTree.derivative.differentiate(qc_d, p[0])
        val_dd = OpTree.evaluate.evaluate_with_estimator(qc_dd, operator, p_array, {},Estimator())

        assert np.linalg.norm(np.abs(np.gradient(val, p_val)[1:-1]-val_d[1:-1])) < 0.15
        assert np.linalg.norm(np.abs(np.gradient(val, p_val)[2:-2]-val_d[2:-2])) < 1.5


    def test_gradient(self):

        # set-up of the expectation value
        p = ParameterVector('p', 4)
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(2)
        qc.rx(p[0]*np.arccos(x[0]), 0)
        qc.rx(p[1]*np.arccos(x[0]), 1)
        qc.ry(p[2],0)
        qc.ry(p[3],1)
        qc.cx(0, 1)
        qc.rx(p[0]*np.arccos(x[0]), 0)
        qc.rx(p[1]*np.arccos(x[0]), 1)
        qc.ry(p[2],0)
        qc.ry(p[3],1)
        qc.cx(0, 1)
        operator = SparsePauliOp(["IZ", "ZI"])
        dictionary = {x[0]:0.5, p[0]:1.5, p[1]:2.5, p[2]:0.5, p[3]:0.25}

        qc_grad = OpTree.derivative.differentiate(qc, p)
        qc_grad_v2 = OpTree.derivative.differentiate_v2(qc, p)
        reference_grad = np.array([-0.89953244,  0.59256683, -0.94626838,  1.73757048])
        assert np.allclose(OpTree.evaluate.evaluate_with_estimator(qc_grad, operator, dictionary, {},Estimator()), reference_grad)
        assert np.allclose(OpTree.evaluate.evaluate_with_estimator(qc_grad_v2, operator, dictionary, {},Estimator()), reference_grad)

        qc_dx = OpTree.derivative.differentiate(qc, x)
        qc_dx_v2 = OpTree.derivative.differentiate_v2(qc, x)
        reference_dx = np.array([-0.14568138])
        assert np.allclose(OpTree.evaluate.evaluate_with_estimator(qc_dx, operator, dictionary, {},Estimator()), reference_dx)
        assert np.allclose(OpTree.evaluate.evaluate_with_estimator(qc_dx_v2, operator, dictionary, {},Estimator()), reference_dx)

        tree = OpTreeList([OpTree.gen_expectation_tree(qc, operator)])
        tree_grad = OpTree.derivative.differentiate(tree, p)
        tree_grad_v2 = OpTree.derivative.differentiate_v2(tree, p)
        assert np.allclose(OpTree.evaluate.evaluate_tree_with_estimator(tree_grad, dictionary, Estimator()), reference_grad)
        assert np.allclose(OpTree.evaluate.evaluate_tree_with_estimator(tree_grad_v2, dictionary, Estimator()), reference_grad)
