import pytest
import numpy as np

from qiskit import QuantumCircuit

from squlearn.feature_map import LayeredFeatureMap





class TestLayeredFeatureMap:


    def test_layered_feature_map_gates():
        # Create a LayeredFeatureMap with 2 layers and 3 features per layer
        lfm = LayeredFeatureMap(num_layers=2, num_features=3)
        
        # Test the H gate
        lfm.H()
        expected_circuit = QuantumCircuit(6)
        expected_circuit.h(range(6))
        assert lfm.get_circuit().draw() == expected_circuit.draw()
        
        # Test the X gate
        lfm.X()
        expected_circuit.x(range(6))
        assert lfm.get_circuit().draw() == expected_circuit.draw()
        
        # Test the Y gate
        lfm.Y()
        expected_circuit.y(range(6))
        assert lfm.get_circuit().draw() == expected_circuit.draw()
        
        # Test the Z gate
        lfm.Z()
        expected_circuit.z(range(6))
        assert lfm.get_circuit().draw() == expected_circuit.draw()
        
        # Test the I gate
        lfm.I()
        expected_circuit.i(range(6))
        assert lfm.get_circuit().draw() == expected_circuit.draw()
        
        # Test the S gate
        lfm.S()
        expected_circuit.s(range(6))
        assert lfm.get_circuit().draw() == expected_circuit.draw()
        
        # Test the T gate
        lfm.T()
        expected_circuit.t(range(6))
        assert lfm.get_circuit().draw() == expected_circuit.draw()