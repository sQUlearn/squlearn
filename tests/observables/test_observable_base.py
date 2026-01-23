import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from squlearn.observables.observable_base import ObservableBase


class DummyOb(ObservableBase):

    def __init__(self, num_qubits, num_parameters=1):
        super().__init__(num_qubits=num_qubits)
        self._num_parameters = num_parameters

    def get_pauli(self, parameters):
        return SparsePauliOp.from_list([("ZZ", 1.0)])

    @property
    def num_parameters(self):
        return self._num_parameters


class TestObservableBase:

    @pytest.fixture
    def ob(self):
        return DummyOb(num_qubits=2)

    def test_init(self, ob: DummyOb):
        assert ob.num_qubits == 2
        assert ob._num_all_qubits == 2
        assert np.array_equal(ob._qubit_map, np.array([0, 1]))
        assert ob.is_mapped is False

    def test_num_parameters(self, ob: DummyOb):
        assert ob.num_parameters == 1

    def test_parameter_bounds(self, ob: DummyOb):
        assert np.array_equal(ob.parameter_bounds, np.array([[0, 5]]))

    def test_set_map(self, ob: DummyOb):
        ob.set_map(qubit_map=[1, 0], num_all_qubits=2)
        assert ob.is_mapped is True
        assert np.array_equal(ob._qubit_map, np.array([1, 0]))
        assert ob._num_all_qubits == 2

        with pytest.raises(ValueError):
            ob.set_map(qubit_map=[2, 0], num_all_qubits=0)

    def test_generate_initial_parameters(self):
        ob = DummyOb(num_qubits=2)
        params = ob.generate_initial_parameters(ones=True, seed=42)
        assert np.array_equal(params, np.array([1]))

        params = ob.generate_initial_parameters(ones=False, seed=42)
        assert params.shape == (1,)

        ob = DummyOb(num_qubits=2, num_parameters=0)
        params = ob.generate_initial_parameters(ones=False, seed=42)
        assert np.array_equal(params, np.array([]))

    def test_get_pauli_mapped(self, ob: DummyOb):
        mapped = ob.get_pauli_mapped([0, 1])
        assert isinstance(mapped, SparsePauliOp)
        assert mapped.num_qubits == 2
        assert mapped.to_list() == [("ZZ", 1.0)]


class TestAddedObservable:

    def test_add_raises(self):
        ob1 = DummyOb(num_qubits=2)

        with pytest.raises(ValueError):
            ob1 + 5

        with pytest.raises(ValueError):
            ob3 = DummyOb(num_qubits=3)
            ob1 + ob3

    def test_add_diff_obs(self):
        ob1 = DummyOb(num_qubits=2)
        ob2 = DummyOb(num_qubits=2)
        ob_sum = ob1 + ob2
        assert isinstance(ob_sum, ObservableBase)
        assert ob_sum.num_qubits == 2
        assert ob_sum.num_parameters == ob1.num_parameters + ob2.num_parameters

    def test_diff_obs_set_params(self):
        ob1 = DummyOb(num_qubits=2)
        ob2 = DummyOb(num_qubits=2)
        ob_sum = ob1 + ob2
        ob_sum.set_params(num_qubits=3)
        assert ob_sum._op1.num_qubits == 3 and ob_sum._op2.num_qubits == 3

    def test_diff_obs_get_params(self):
        ob1 = DummyOb(num_qubits=2)
        ob2 = DummyOb(num_qubits=2)
        ob_sum = ob1 + ob2
        params = ob_sum.get_params()

        assert params["op1"] is ob1
        assert params["op2"] is ob2

        assert params["num_qubits"] == ob1.get_params()["num_qubits"]

        for k, val in ob1.get_params().items():
            if k != "num_qubits":
                assert params[f"op1__{k}"] == val

        for k, val in ob2.get_params().items():
            if k != "num_qubits":
                assert params[f"op2__{k}"] == val

    def test_diff_obs_parameter_bounds(self):
        ob1 = DummyOb(num_qubits=2, num_parameters=2)
        ob2 = DummyOb(num_qubits=2, num_parameters=3)
        ob_sum = ob1 + ob2
        bounds = ob_sum.parameter_bounds
        assert bounds.shape == (5, 2)
        assert np.array_equal(bounds[:2], ob1.parameter_bounds)
        assert np.array_equal(bounds[2:], ob2.parameter_bounds)

    def test_same_obs(self):
        ob1 = DummyOb(num_qubits=2)
        ob_sum = ob1 + ob1
        assert isinstance(ob_sum, ObservableBase)
        assert ob_sum.num_qubits == 2
        assert ob_sum.num_parameters == ob1.num_parameters

    def test_same_obs_set_params(self):
        ob1 = DummyOb(num_qubits=2)
        ob_sum = ob1 + ob1
        ob_sum.set_params(num_qubits=3)
        assert ob_sum._op1.num_qubits == 3 and ob_sum._op2.num_qubits == 3

    def test_same_obs_get_params(self):
        ob1 = DummyOb(num_qubits=2)
        ob_sum = ob1 + ob1
        assert ob_sum.get_params() == ob1.get_params()

    def test_same_obs_parameter_bounds(self):
        ob1 = DummyOb(num_qubits=2, num_parameters=2)
        ob_sum = ob1 + ob1
        bounds = ob_sum.parameter_bounds
        assert bounds.shape == (2, 2)
        assert np.array_equal(bounds, ob1.parameter_bounds)

    def test_generate_initial_parameters(self):
        ob1 = DummyOb(num_qubits=2, num_parameters=2)
        ob2 = DummyOb(num_qubits=2, num_parameters=2)
        ob_sum = ob1 + ob2

        params = ob_sum.generate_initial_parameters(ones=True, seed=42)
        assert np.array_equal(params, np.array([1, 1, 1, 1]))


class TestMultipliedObservable:

    def test_mult_raises(self):
        ob1 = DummyOb(num_qubits=2)

        with pytest.raises(ValueError):
            ob1 * 5

        with pytest.raises(ValueError):
            ob3 = DummyOb(num_qubits=3)
            ob1 * ob3

    def test_mult_diff_obs(self):
        ob1 = DummyOb(num_qubits=2)
        ob2 = DummyOb(num_qubits=2)
        ob_prod = ob1 * ob2
        assert isinstance(ob_prod, ObservableBase)
        assert ob_prod.num_qubits == 2
        assert ob_prod.num_parameters == ob1.num_parameters + ob2.num_parameters

    def test_diff_obs_set_params(self):
        ob1 = DummyOb(num_qubits=2)
        ob2 = DummyOb(num_qubits=2)
        ob_prod = ob1 * ob2
        ob_prod.set_params(num_qubits=3)
        assert ob_prod._op1.num_qubits == 3 and ob_prod._op2.num_qubits == 3

    def test_diff_obs_get_params(self):
        ob1 = DummyOb(num_qubits=2)
        ob2 = DummyOb(num_qubits=2)
        ob_prod = ob1 * ob2
        params = ob_prod.get_params()

        assert params["op1"] is ob1
        assert params["op2"] is ob2

        assert params["num_qubits"] == ob1.get_params()["num_qubits"]

        for k, val in ob1.get_params().items():
            if k != "num_qubits":
                assert params[f"op1__{k}"] == val

        for k, val in ob2.get_params().items():
            if k != "num_qubits":
                assert params[f"op2__{k}"] == val

    def test_same_obs(self):
        ob1 = DummyOb(num_qubits=2)
        ob_prod = ob1 * ob1
        assert isinstance(ob_prod, ObservableBase)
        assert ob_prod.num_qubits == 2
        assert ob_prod.num_parameters == ob1.num_parameters

    def test_same_obs_set_params(self):
        ob1 = DummyOb(num_qubits=2)
        ob_prod = ob1 * ob1
        ob_prod.set_params(num_qubits=3)
        assert ob_prod._op1.num_qubits == 3 and ob_prod._op2.num_qubits == 3

    def test_same_obs_get_params(self):
        ob1 = DummyOb(num_qubits=2)
        ob_prod = ob1 * ob1
        assert ob_prod.get_params() == ob1.get_params()
