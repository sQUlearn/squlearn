#!/usr/bin/env python3
"""Test script to verify read-only properties for encoding circuits."""

import sys
from squlearn.encoding_circuit import (
    MultiControlEncodingCircuit,
    ChebyshevTower,
    RandomLayeredEncodingCircuit,
    HighDimEncodingCircuit,
)


def test_multi_control_encoding_circuit():
    """Test MultiControlEncodingCircuit properties."""
    print("Testing MultiControlEncodingCircuit...")
    circuit = MultiControlEncodingCircuit(
        num_qubits=4, num_layers=2, closed=False, final_encoding=True
    )
    
    # Test properties are readable
    assert circuit.num_layers == 2, f"Expected num_layers=2, got {circuit.num_layers}"
    assert circuit.closed == False, f"Expected closed=False, got {circuit.closed}"
    assert circuit.final_encoding == True, f"Expected final_encoding=True, got {circuit.final_encoding}"
    
    # Test get_params
    params = circuit.get_params()
    assert params["num_layers"] == 2
    assert params["closed"] == False
    assert params["final_encoding"] == True
    
    # Test set_params
    circuit.set_params(num_layers=3, closed=True, final_encoding=False)
    assert circuit.num_layers == 3
    assert circuit.closed == True
    assert circuit.final_encoding == False
    
    # Test properties are read-only
    try:
        circuit.num_layers = 5
        print("  ERROR: num_layers should be read-only!")
        return False
    except AttributeError:
        pass  # Expected
    
    print("  ✓ MultiControlEncodingCircuit passed")
    return True


def test_chebyshev_tower():
    """Test ChebyshevTower properties."""
    print("Testing ChebyshevTower...")
    circuit = ChebyshevTower(
        num_qubits=4,
        num_chebyshev=3,
        alpha=2.0,
        num_layers=2,
        rotation_gate="rx",
        hadamard_start=False,
        arrangement="alternating",
        nonlinearity="arctan",
    )
    
    # Test properties are readable
    assert circuit.num_chebyshev == 3, f"Expected num_chebyshev=3, got {circuit.num_chebyshev}"
    assert circuit.alpha == 2.0, f"Expected alpha=2.0, got {circuit.alpha}"
    assert circuit.num_layers == 2, f"Expected num_layers=2, got {circuit.num_layers}"
    assert circuit.rotation_gate == "rx", f"Expected rotation_gate='rx', got {circuit.rotation_gate}"
    assert circuit.hadamard_start == False, f"Expected hadamard_start=False, got {circuit.hadamard_start}"
    assert circuit.arrangement == "alternating", f"Expected arrangement='alternating', got {circuit.arrangement}"
    assert circuit.nonlinearity == "arctan", f"Expected nonlinearity='arctan', got {circuit.nonlinearity}"
    
    # Test get_params
    params = circuit.get_params()
    assert params["num_chebyshev"] == 3
    assert params["alpha"] == 2.0
    assert params["num_layers"] == 2
    assert params["rotation_gate"] == "rx"
    assert params["hadamard_start"] == False
    assert params["arrangement"] == "alternating"
    assert params["nonlinearity"] == "arctan"
    
    # Test set_params
    circuit.set_params(num_chebyshev=4, alpha=1.5, num_layers=3)
    assert circuit.num_chebyshev == 4
    assert circuit.alpha == 1.5
    assert circuit.num_layers == 3
    
    # Test properties are read-only
    try:
        circuit.num_layers = 5
        print("  ERROR: num_layers should be read-only!")
        return False
    except AttributeError:
        pass  # Expected
    
    print("  ✓ ChebyshevTower passed")
    return True


def test_random_layered_encoding_circuit():
    """Test RandomLayeredEncodingCircuit properties."""
    print("Testing RandomLayeredEncodingCircuit...")
    circuit = RandomLayeredEncodingCircuit(
        num_qubits=4,
        seed=42,
        min_num_layers=3,
        max_num_layers=8,
        feature_probability=0.5,
    )
    
    # Test properties are readable
    assert circuit.seed == 42, f"Expected seed=42, got {circuit.seed}"
    assert circuit.min_num_layers == 3, f"Expected min_num_layers=3, got {circuit.min_num_layers}"
    assert circuit.max_num_layers == 8, f"Expected max_num_layers=8, got {circuit.max_num_layers}"
    assert circuit.feature_probability == 0.5, f"Expected feature_probability=0.5, got {circuit.feature_probability}"
    
    # Test get_params
    params = circuit.get_params()
    assert params["seed"] == 42
    assert params["min_num_layers"] == 3
    assert params["max_num_layers"] == 8
    assert params["feature_probability"] == 0.5
    
    # Test set_params
    circuit.set_params(seed=100, min_num_layers=2, max_num_layers=10)
    assert circuit.seed == 100
    assert circuit.min_num_layers == 2
    assert circuit.max_num_layers == 10
    
    # Test properties are read-only
    try:
        circuit.seed = 999
        print("  ERROR: seed should be read-only!")
        return False
    except AttributeError:
        pass  # Expected
    
    print("  ✓ RandomLayeredEncodingCircuit passed")
    return True


def test_highdim_encoding_circuit():
    """Test HighDimEncodingCircuit properties."""
    print("Testing HighDimEncodingCircuit...")
    circuit = HighDimEncodingCircuit(
        num_qubits=5,
        cycling=False,
        cycling_type="hat",
        num_layers=3,
        layer_type="columns",
        entangling_gate="cx",
    )
    
    # Test properties are readable
    assert circuit.cycling == False, f"Expected cycling=False, got {circuit.cycling}"
    assert circuit.cycling_type == "hat", f"Expected cycling_type='hat', got {circuit.cycling_type}"
    assert circuit.num_layers == 3, f"Expected num_layers=3, got {circuit.num_layers}"
    assert circuit.layer_type == "columns", f"Expected layer_type='columns', got {circuit.layer_type}"
    assert circuit.entangling_gate == "cx", f"Expected entangling_gate='cx', got {circuit.entangling_gate}"
    
    # Test get_params
    params = circuit.get_params()
    assert params["cycling"] == False
    assert params["cycling_type"] == "hat"
    assert params["num_layers"] == 3
    assert params["layer_type"] == "columns"
    assert params["entangling_gate"] == "cx"
    
    # Test set_params
    circuit.set_params(cycling=True, num_layers=4)
    assert circuit.cycling == True
    assert circuit.num_layers == 4
    
    # Test properties are read-only
    try:
        circuit.num_layers = 5
        print("  ERROR: num_layers should be read-only!")
        return False
    except AttributeError:
        pass  # Expected
    
    print("  ✓ HighDimEncodingCircuit passed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Read-Only Properties for Encoding Circuits")
    print("=" * 60)
    
    results = []
    results.append(test_multi_control_encoding_circuit())
    results.append(test_chebyshev_tower())
    results.append(test_random_layered_encoding_circuit())
    results.append(test_highdim_encoding_circuit())
    
    print("=" * 60)
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
