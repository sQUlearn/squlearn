""" Tests for OpTree evaluation"""

import pytest
from typing import Tuple, List
import numpy as np
from packaging import version

from qiskit.circuit import QuantumCircuit
from qiskit import __version__ as qiskit_version
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit
from qiskit.circuit import ParameterVector

from squlearn.util import OpTree
from squlearn.util.optree import OpTreeSum, OpTreeList

QISKIT_SMALLER_1_0 = version.parse(qiskit_version) < version.parse("1.0.0")


class TestOpTreeEvaluation:
    """Test class for OpTree evaluation"""

    @pytest.fixture(scope="module")
    def _create_random_circuits(self) -> OpTreeList:
        """Creates the random circuits used in the tests"""
        circuit1 = QuantumCircuit(2)
        circuit1.h(0)
        circuit1.cx(0, 1)
        circuit1.rz(0.5, 0)
        circuit1.ry(0.5, 1)
        circuit2 = QuantumCircuit(2)
        circuit2.h(0)
        circuit2.cp(0.5, 0, 1)
        circuit2.s(0)
        circuit2.t(1)
        circuit2.h(0)
        circuit2.h(1)
        return OpTreeList([circuit1, circuit2])

    @pytest.fixture(scope="module")
    def _create_param_circuits(self) -> Tuple[OpTreeList, List[dict]]:
        p = ParameterVector("p", 2)
        circuit1 = QuantumCircuit(2)
        circuit1.rx(p[0], 0)
        circuit1.rx(p[1], 1)
        circuit2 = QuantumCircuit(2)
        circuit2.ry(p[0], 0)
        circuit2.ry(p[1], 1)
        dictionary1 = {p[0]: 0.25, p[1]: 0.5}
        dictionary2 = {p[0]: 0.33, p[1]: 0.44}
        return OpTreeList([circuit1, circuit2]), [dictionary1, dictionary2]

    @pytest.fixture(scope="module")
    def _create_operator_z(self) -> Tuple[OpTreeSum, List[dict]]:
        """Creates the Z-based operators used in the tests"""
        x = ParameterVector("x", 2)
        observable1 = SparsePauliOp(["IZ", "ZI"], [x[0], x[1]])
        observable2 = SparsePauliOp(["II", "ZZ"], [x[0], x[1]])
        observable = OpTreeSum([observable1, observable2])
        dictionary1 = {x[0]: 1.0, x[1]: 0.5}
        dictionary2 = {x[0]: 0.3, x[1]: 0.2}
        return observable, [dictionary1, dictionary2]

    @pytest.fixture(scope="module")
    def _create_operator_xy(self) -> Tuple[OpTreeSum, dict]:
        """Creates the XY-based operators used in the tests"""
        x = ParameterVector("x", 2)
        observable1 = SparsePauliOp(["XY", "YX"], [x[0], x[1]])
        observable2 = SparsePauliOp(["ZZ", "YY"], [x[0], x[1]])
        observable = OpTreeSum([observable1, observable2])
        dictionary = {x[0]: 1.0, x[1]: 0.5}
        return observable, dictionary

    def test_estimator_z(self, _create_random_circuits, _create_operator_z):
        """Tests the estimator with Z basis operators

        Args:
            _create_random_circuits (Tuple[OpTreeList, List[dict]]): The circuits and dictionaries.
            _create_operator_z (Tuple[OpTreeSum, List[dict]]): The operators and dictionaries.
        """

        if QISKIT_SMALLER_1_0:
            from qiskit.primitives import Estimator

            estimator = Estimator()
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator(default_precision=0)

        reference_values = np.array([1.43879128, 1.0])

        # Check functionality of estimator evaluation
        val = OpTree.evaluate.evaluate_with_estimator(
            _create_random_circuits,
            _create_operator_z[0],
            {},
            _create_operator_z[1][0],
            estimator=estimator,
        )
        assert np.allclose(val, reference_values)

        # Check functionality of estimator tree evaluation
        expectation_tree = OpTree.gen_expectation_tree(
            _create_random_circuits, _create_operator_z[0]
        )
        val = OpTree.evaluate.evaluate_tree_with_estimator(
            expectation_tree, _create_operator_z[1][0], estimator=estimator
        )
        assert np.allclose(val, reference_values)

    def test_sampler_z(self, _create_random_circuits, _create_operator_z):
        """Tests the sampler with Z basis operators

        Args:
            _create_random_circuits (Tuple[OpTreeList, List[dict]]): The circuits and dictionaries.
            _create_operator_z (Tuple[OpTreeSum, List[dict]]): The operators and dictionaries.
        """

        if QISKIT_SMALLER_1_0:
            from qiskit.primitives import Sampler

            sampler = Sampler()
            reference_values = np.array([1.43879128, 1.0])
        else:
            from qiskit.primitives import StatevectorSampler

            sampler = StatevectorSampler(seed=0, default_shots=5000)
            # StatevectorSampler does only support sampling, not statevectors
            reference_values = np.array([1.4524, 1.0102])

        # Check functionality of sampler evaluation
        val = OpTree.evaluate.evaluate_with_sampler(
            _create_random_circuits, _create_operator_z[0], {}, _create_operator_z[1][0], sampler
        )
        assert np.allclose(val, reference_values)

        # Check functionality of sampler tree evaluation
        expectation_tree = OpTree.gen_expectation_tree(
            _create_random_circuits, _create_operator_z[0]
        )
        val = OpTree.evaluate.evaluate_tree_with_sampler(
            expectation_tree, _create_operator_z[1][0], sampler
        )
        assert np.allclose(val, reference_values)

    def test_estimator_xy(self, _create_random_circuits, _create_operator_xy):
        """
        Tests the estimator with Z basis operators

        Args:
            _create_random_circuits (Tuple[OpTreeList, List[dict]]): The circuits and dictionaries.
            _create_operator_xy (Tuple[OpTreeSum, dict]): The operators and dictionary.
        """

        if QISKIT_SMALLER_1_0:
            from qiskit.primitives import Estimator

            estimator = Estimator()
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator(default_precision=0)

        reference_values = np.array([1.09923954, -1.0])

        # Check functionality of estimator evaluation
        val = OpTree.evaluate.evaluate_with_estimator(
            _create_random_circuits,
            _create_operator_xy[0],
            {},
            _create_operator_xy[1],
            estimator,
        )
        assert np.allclose(val, reference_values)

        # Check functionality of estimator tree evaluation
        expectation_tree = OpTree.gen_expectation_tree(
            _create_random_circuits, _create_operator_xy[0]
        )
        val = OpTree.evaluate.evaluate_tree_with_estimator(
            expectation_tree, _create_operator_xy[1], estimator
        )
        assert np.allclose(val, reference_values)

    def test_sampler_xy(self, _create_random_circuits, _create_operator_xy):
        """Tests the estimator with Z basis operators

        Args:
            _create_random_circuits (Tuple[OpTreeList, List[dict]]): The circuits and dictionaries.
            _create_operator_xy (Tuple[OpTreeSum, dict]): The operators and dictionary.
        """

        if QISKIT_SMALLER_1_0:
            from qiskit.primitives import Sampler

            sampler = Sampler()
            reference_values = np.array([1.09923954, -1.0])
        else:
            from qiskit.primitives import StatevectorSampler

            sampler = StatevectorSampler(seed=0, default_shots=5000)
            # StatevectorSampler does only support sampling, not statevectors
            reference_values = np.array([1.1204, -0.9738])

        # Check functionality of evaluation
        with pytest.raises(ValueError):
            OpTree.evaluate.evaluate_with_sampler(
                _create_random_circuits,
                _create_operator_xy[0],
                {},
                _create_operator_xy[1],
                sampler,
            )
        op_in_z_base = OpTree.evaluate.transform_to_zbasis(_create_operator_xy[0])
        val = OpTree.evaluate.evaluate_with_sampler(
            _create_random_circuits, op_in_z_base, {}, _create_operator_xy[1], sampler
        )
        assert np.allclose(val, reference_values)

        # Check functionality of tree evaluation
        expectation_tree = OpTree.gen_expectation_tree(
            _create_random_circuits, _create_operator_xy[0]
        )
        with pytest.raises(ValueError):
            OpTree.evaluate.evaluate_tree_with_sampler(
                expectation_tree, _create_operator_xy[1], sampler
            )
        expectation_tree_in_z_base = OpTree.evaluate.transform_to_zbasis(expectation_tree)
        val = OpTree.evaluate.evaluate_tree_with_sampler(
            expectation_tree_in_z_base, _create_operator_xy[1], sampler
        )
        assert np.allclose(val, reference_values)

    def test_estimator_multi_dict(self, _create_param_circuits, _create_operator_z):
        """
        Checks the functionality of the estimator with multiple dictionaries.

        Args:
            _create_param_circuits (Tuple[OpTreeList, List[dict]]): The circuits and dictionaries.
            _create_operator_z (Tuple[OpTreeSum, List[dict]]): The operators and dictionaries.
        """

        if QISKIT_SMALLER_1_0:
            from qiskit.primitives import Estimator

            estimator = Estimator()
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator(default_precision=0)

        reference_values = np.array(
            [
                [[2.83285403, 2.83285403], [0.93625037, 0.93625037]],
                [[2.82638487, 2.82638487], [0.93594971, 0.93594971]],
            ]
        )
        val = OpTree.evaluate.evaluate_with_estimator(
            _create_param_circuits[0],
            _create_operator_z[0],
            _create_param_circuits[1],
            _create_operator_z[1],
            estimator,
        )
        assert np.allclose(val, reference_values)

        reference_values = np.array([[2.83285403, 2.83285403], [0.93594971, 0.93594971]])
        val = OpTree.evaluate.evaluate_with_estimator(
            _create_param_circuits[0],
            _create_operator_z[0],
            _create_param_circuits[1],
            _create_operator_z[1],
            estimator,
            dictionaries_combined=True,
        )
        assert np.allclose(val, reference_values)

    def test_sampler_multi_dict(self, _create_param_circuits, _create_operator_z):
        """
        Checks the functionality of the sampler with multiple dictionaries.

        Args:
            _create_param_circuits (Tuple[OpTreeList, List[dict]]): The circuits and dictionaries.
            _create_operator_z (Tuple[OpTreeSum, List[dict]]): The operators and dictionaries.

        """

        if QISKIT_SMALLER_1_0:
            from qiskit.primitives import Sampler

            sampler = Sampler()
            reference_values = np.array(
                [
                    [[2.83285403, 2.83285403], [0.93625037, 0.93625037]],
                    [[2.82638487, 2.82638487], [0.93594971, 0.93594971]],
                ]
            )
            reference_values2 = np.array([[2.83285403, 2.83285403], [0.93594971, 0.93594971]])
        else:
            from qiskit.primitives import StatevectorSampler

            sampler = StatevectorSampler(seed=0, default_shots=5000)
            # StatevectorSampler does only support sampling, not statevectors
            reference_values = np.array(
                [[[2.8344, 2.8344], [0.9368, 0.9368]], [[2.8264, 2.8264], [0.93608, 0.93608]]]
            )
            reference_values2 = np.array([[2.8344, 2.8344], [0.93608, 0.93608]])

        val = OpTree.evaluate.evaluate_with_sampler(
            _create_param_circuits[0],
            _create_operator_z[0],
            _create_param_circuits[1],
            _create_operator_z[1],
            sampler,
        )
        assert np.allclose(val, reference_values)

        val = OpTree.evaluate.evaluate_with_sampler(
            _create_param_circuits[0],
            _create_operator_z[0],
            _create_param_circuits[1],
            _create_operator_z[1],
            sampler,
            dictionaries_combined=True,
        )
        assert np.allclose(val, reference_values2)
