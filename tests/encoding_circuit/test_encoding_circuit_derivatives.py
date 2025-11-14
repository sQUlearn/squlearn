import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit, ParameterVector
from squlearn.encoding_circuit import EncodingCircuitDerivatives
from squlearn.util.optree.optree import OpTreeCircuit, OpTreeList, OpTreeElementBase, OpTreeSum


class DummyEncodingCircuit:
    def __init__(self, num_qubits: int, num_parameters: int):
        self.num_qubits = num_qubits
        self.num_parameters = num_parameters

    def get_circuit(
        self, features: ParameterVector, parameters: ParameterVector
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # if features / parameters contain entries, use the first of each to create parameterized gates
        if len(features) > 0:
            qc.rz(features[0], 0 % self.num_qubits)
        if len(parameters) > 0:
            # use a different qubit index if possible
            qc.rx(parameters[0], (0 if self.num_qubits == 1 else 1))
        return qc


class TestEncodingCircuitDerivatives:

    @pytest.fixture
    def ecd(self) -> EncodingCircuitDerivatives:
        enc = DummyEncodingCircuit(num_qubits=2, num_parameters=1)
        return EncodingCircuitDerivatives(enc, num_features=2, optree_caching=True)

    @pytest.mark.parametrize("optree_caching", [True, False])
    def test_init(self, optree_caching):
        enc = DummyEncodingCircuit(num_qubits=2, num_parameters=1)
        ecd = EncodingCircuitDerivatives(enc, num_features=2, optree_caching=optree_caching)

        assert ecd.num_features == 2
        assert ecd.num_parameters == 1
        assert ecd.num_qubits == 2

        # Check that the starting circuit is correctly wrapped as an OpTreeCircuit
        assert isinstance(ecd._optree_start, OpTreeCircuit)

        # Check that the parameters are correctly assigned
        assert len(ecd._x) == 2
        assert len(ecd._p) == 1
        assert ecd._x[0].name == "x[0]"
        assert ecd._p[0].name == "p[0]"

        # Check that the optree_cache is initialized correctly
        if ecd._optree_caching:
            assert isinstance(ecd._optree_cache, dict)
            assert len(ecd._optree_cache) == 1
        else:
            assert ecd._optree_cache == {}

    @pytest.mark.parametrize(
        "deriv, expected_type",
        [
            ("I", OpTreeCircuit),
            ("dx", OpTreeElementBase),
            ("dxdx", OpTreeElementBase),
            ("dpdxdx", OpTreeElementBase),
            ("laplace", OpTreeSum),
            ("laplace_dp", OpTreeSum),
            ("dp", OpTreeElementBase),
            ("dpdp", OpTreeElementBase),
            ("dpdx", OpTreeElementBase),
            ("dxdp", OpTreeElementBase),
        ],
    )
    def test_get_derivative(self, deriv, expected_type):
        enc = DummyEncodingCircuit(num_qubits=2, num_parameters=1)
        ecd = EncodingCircuitDerivatives(enc, num_features=2, optree_caching=True)
        op_tree = ecd.get_derivative(deriv)
        assert isinstance(op_tree, expected_type)

    def test_assign_parameters_returns_none_for_none_optree(self):
        enc = DummyEncodingCircuit(num_qubits=1, num_parameters=0)
        ecd = EncodingCircuitDerivatives(enc, num_features=1, optree_caching=False)
        res = ecd.assign_parameters(None, features=np.array([0.1]), parameters=np.array([]))
        assert res is None

    def test_assign_parameters_single_feature_and_parameter(self, ecd):
        features = np.array([0.1, 0.2])
        parameters = np.array([0.3])
        res = ecd.assign_parameters(ecd._optree_start, features=features, parameters=parameters)
        assert res is not None
        assert isinstance(res, OpTreeElementBase)
        assert not isinstance(res, OpTreeList)

    def test_optree_differentiation_empty_parameters_returns_empty_optreelist(self):
        enc = DummyEncodingCircuit(num_qubits=1, num_parameters=0)
        ecd = EncodingCircuitDerivatives(enc, num_features=1, optree_caching=False)

        # empty parameter list -> should return OpTreeList([])
        res = ecd._optree_differentiation(ecd._optree_start, parameters=[])
        assert isinstance(res, OpTreeList)
        assert len(res.children) == 0
        assert len(res.operation) == 0
        assert len(res.factor) == 0

    def test_optree_differentiation_mixed_parameter_types_raises_typeerror(self):
        enc = DummyEncodingCircuit(num_qubits=2, num_parameters=1)
        ecd = EncodingCircuitDerivatives(enc, num_features=1, optree_caching=False)

        # ecd._x[0] and ecd._p[0] are ParameterVectorElement objects with different base names ('x' vs 'p')
        with pytest.raises(TypeError, match="Differentiable variables are not the same type."):
            ecd._optree_differentiation(ecd._optree_start, parameters=[ecd._x[0], ecd._p[0]])
