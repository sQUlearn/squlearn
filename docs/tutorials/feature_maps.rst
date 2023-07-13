.. _quantum_feature_maps:

.. currentmodule:: squlearn.feature_map

####################
Quantum Feature Maps
####################

Quantum feature maps are used to embed classical data :math:`x` into a quantum state and manipulate
the quantum state via trainable parameters :math:`p`.
They are a key component of many quantum machine learning algorithms, and the design of a good
quantum feature map is crucial for the performance of the algorithm.
In sQUlearn, feature maps are an obligatory input in the Quantum Neural Network (QNN) or Quantum
Kernel programs.
sQUlearn offers a wide range of pre-implemented quantum feature maps,
which can be combined to create more sophisticated feature maps.
Furthermore, it is possible to create custom feature maps that follow a layered approach, in which
each gate is applied to all qubits.
The package facilitate a fully automated pruning algorithm to remove redundant parameters and
enables the automatic differentiation of arbitrary derivative.

The following functions and classes are are accessible via :class:`squlearn.feature_map`.

Implemented Quantum Feature Maps
-----------------------------------

There are several Quantum Feature Maps implemented in sQUlearn:

.. autosummary::
   :nosignatures:

   YZ_CX_FeatureMap
   HighDimFeatureMap
   QEKFeatureMap
   ChebyshevTower
   ChebPQC
   HZCRxCRyCRz
   ChebRx
   ParamZFeatureMap
   QiskitZFeatureMap

Feel free to contribute to sQulearn by adding your own feature maps in a Pull Request.


**Example: Create a QEK feature map**

.. code-block:: python

   from squlearn.feature_map import QEKFeatureMap
   pqc = QEKFeatureMap(num_qubits=4, num_features=2, num_layers=2)
   pqc.draw()

.. plot::

   from squlearn.feature_map import QEKFeatureMap
   pqc = QEKFeatureMap(num_qubits=4, num_features=2, num_layers=2)
   plt = pqc.draw(style={'fontsize':15,'subfontsize': 10})
   plt.tight_layout()
   plt


Combining Quantum Feature Maps
------------------------------
In sQUlearn, quantum feature maps can be combined to create more sophisticated feature maps by
utilizing the ``+`` operation between two feature maps. However, it is important to note that
the number of qubits in both feature maps must match for successful combination.

When combining feature maps, the resulting feature dimension is determined by taking the maximum
value from the two feature dimensions. The parameters of the individual feature maps are
concatenated. Consequently, the total number of parameters in the combined feature map is
equal to the sum of the parameters in the two original feature maps.

**Example: combine two quantum feature maps**

.. code-block:: python

   from squlearn.feature_map import QEKFeatureMap, ChebPQC
   fm1 = QEKFeatureMap(num_qubits=4, num_features=2, num_layers=1, closed=False)
   fm2 = ChebPQC(num_qubits=4, num_features=3, num_layers=1)
   # Combining both feature maps
   fm3 = fm1 + fm2
   fm3.draw()

.. plot::

   from squlearn.feature_map import QEKFeatureMap, ChebPQC
   fm1 = QEKFeatureMap(num_qubits=4, num_features=2, num_layers=1, closed=False)
   fm2 = ChebPQC(num_qubits=4, num_features=3, num_layers=1)
   # Combining both feature maps
   fm3 = fm1 + fm2
   plt = fm3.draw(style={'fontsize':15,'subfontsize': 10})
   plt.tight_layout()
   plt


Wrapping Qiskit Feature Maps
----------------------------

It is also possible to utilize the wrapper :class:`QiskitFeatureMap` to build Feature Maps from the
`Qiskit circuit library <https://qiskit.org/documentation/apidoc/circuit_library.html>`_.

.. code-block:: python

   from squlearn.feature_map import QiskitFeatureMap
   from qiskit.circuit.library import TwoLocal
   local = TwoLocal(3, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
   QiskitFeatureMap(local).draw()

.. plot::

   from squlearn.feature_map import QiskitFeatureMap
   from qiskit.circuit.library import TwoLocal
   local = TwoLocal(3, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
   pqc = QiskitFeatureMap(local)
   plt = pqc.draw(style={'fontsize':15,'subfontsize': 10})
   plt.tight_layout()
   plt


Create your custom Feature Map via :class:`LayeredFeatureMap`
-------------------------------------------------------------

sQUlearn offers a user-friendly solution for creating custom layered feature maps effortlessly.
Layered feature maps involve the application of gates to all qubits, ensuring a comprehensive
approach. This method allows for the creation of feature maps in a structured manner,
regardless of the number of qubits involved.
Two-qubit gates are applied either in a nearest neighbor fashion or by entangling all qubits.
You can construct the layered feature map using either a Qiskit Quantum circuit-inspired approach
or by providing a string using the
:meth:`LayeredFeatureMap.from_string() <squlearn.feature_map.LayeredFeatureMap.from_string>`
method. For detailed instructions on the string format, please refer to the documentation of
the :class:`LayeredFeatureMap` class.

**Example: Create your custom layered feature map**

.. code-block:: python

   from squlearn.feature_map import LayeredFeatureMap
   from squlearn.feature_map.layered_feature_map import Layer
   feature_map = LayeredFeatureMap(num_qubits=4,num_features=2)
   feature_map.H()
   layer = Layer(feature_map)
   layer.Rz("x")
   layer.Ry("p")
   layer.cx_entangling("NN")
   feature_map.add_layer(layer,num_layers=3)
   feature_map.draw()

.. plot::

   from squlearn.feature_map import LayeredFeatureMap
   from squlearn.feature_map.layered_feature_map import Layer
   feature_map = LayeredFeatureMap(num_qubits=4,num_features=2)
   feature_map.H()
   layer = Layer(feature_map)
   layer.Rz("x")
   layer.Ry("p")
   layer.cx_entangling("NN")
   feature_map.add_layer(layer,num_layers=3)
   plt = feature_map.draw(style={'fontsize':15,'subfontsize': 10})
   plt.tight_layout()
   plt

**Example: Create your custom layered feature map from a string**

.. code-block:: python

   from squlearn.feature_map import LayeredFeatureMap
   feature_map = LayeredFeatureMap.from_string(
      "Ry(p)-3[Rx(p,x;=y*np.arccos(x),{y,x})-crz(p)]-Ry(p)", num_qubits=4, num_features=1
   )
   feature_map.draw()

.. plot::

   from squlearn.feature_map import LayeredFeatureMap
   feature_map = LayeredFeatureMap.from_string(
      "Ry(p)-3[Rx(p,x;=y*np.arccos(x),{y,x})-crz(p)]-Ry(p)", num_qubits=4, num_features=1
   )
   plt = feature_map.draw(style={'fontsize':15,'subfontsize': 10})
   plt.tight_layout()
   plt

Pruning of Quantum Feature Maps
-------------------------------

It is also possible to remove parameterized gates from a quantum feature map by using the
:class:`PrunedFeatureMap` class.
This class accepts a quantum feature map as input and removes the parameterized gates
from the feature map for the parameters which indices are specified in the supplied list.
The pruned feature map automatically adjusts the number of parameters and features.

Furthermore it is possible to determine the redundat parameters in feature map automatically.
The algorithm is based on https://doi.org/10.1103/PRXQuantum.2.040309 and is based on evaluating
the Quantum Fisher Information Matrix (QFIM) of the feature map.

sQUlearn features a fully automated pruning algorithm which can be used by calling the routine
:meth:`automated_pruning` that returns a pruned feature map without the redundant parameters.

**Example: Pruning a feature map with redundant parameters**

.. code-block:: python

   from squlearn.feature_map import LayeredFeatureMap, automated_pruning
   from squlearn.util import Executor
   feature_map = LayeredFeatureMap.from_string("Rz(p)-Ry(p)-Z-Ry(p)-Rz(p)", num_qubits=2, num_features=0)
   pruned_feature_map = automated_pruning(feature_map, Executor("statevector_simulator"))
   pruned_feature_map.draw()

.. plot::

   from squlearn.feature_map import LayeredFeatureMap, automated_pruning
   from squlearn.util import Executor
   feature_map = LayeredFeatureMap.from_string("Rz(p)-Ry(p)-Z-Ry(p)-Rz(p)", num_qubits=2, num_features=0)
   pruned_feature_map = automated_pruning(feature_map, Executor("statevector_simulator"))
   plt = pruned_feature_map.draw(style={'fontsize':15,'subfontsize': 10})
   plt.tight_layout()
   plt

Different Quantum Feature Maps via :class:`FeatureMapDerivatives`
-----------------------------------------------------------------

The calculation of derivatives for quantum feature maps is often essential in training a
Quantum Machine Learning model. In sQUlearn, we offer a straightforward approach to compute
these derivatives using the :class:`FeatureMapDerivatives` class.
This class accepts an existing quantum feature map as input and generates derivatives of the
feature map with respect to its parameters or features. The derivative circuits are generated
by leveraging the parameter-shift rule and are cached for future use.
To obtain the derivative, you have two options. Firstly, you can use the function
:meth:`get_derivative() <squlearn.feature_map.FeatureMapDerivatives.get_derivative>`
by supplying a string (a list of available strings can be found in the
:class:`FeatureMapDerivatives` class). Alternatively, you can provide a tuple containing
the parameter (or feature) vector. Additional parameters can be included in the tuple to
obtain arbitrary derivatives.
Currently, the derivatives are returned as a Qiskit Opflow object, but will be converted to
a custom format in the future.

**Example: Obtain the derivative of a QEK feature map**

.. code-block:: python

   from squlearn.feature_map import QEKFeatureMap, FeatureMapDerivatives
   fm = QEKFeatureMap(num_qubits=2, num_features=2, num_layers=2)
   fm_deriv = FeatureMapDerivatives(fm)
   grad_from_string = fm_deriv.get_derivative("dp")
   grad_from_tuple = fm_deriv.get_derivative((fm_deriv.parameter_vector,))


Transpile Quantum Feature Maps via :class:`TranspiledFeatureMap`
----------------------------------------------------------------

To transpile a quantum feature map, you can leverage the functionality provided by the
:class:`TranspiledFeatureMap` class. By utilizing this class, you can input an existing
quantum feature map and have its circuit transpiled according to the specified backend and
transpiler settings, which are the same settings used in Qiskit.
The transpiled feature map is internally employed in the QNN program and projected kernels,
where it is employed internally.

**Example: Transpile a existing Feature Map to a fake backend**

.. code-block:: python

   from squlearn.feature_map import TranspiledFeatureMap,ChebRx
   from qiskit.providers.fake_provider import FakeManilaV2

   fm = TranspiledFeatureMap(ChebRx(3,1),backend=FakeManilaV2(),initial_layout=[0,1,4])
   fm.draw()

.. plot::

   from squlearn.feature_map import TranspiledFeatureMap,ChebRx
   from qiskit.providers.fake_provider import FakeManilaV2
   fm = TranspiledFeatureMap(ChebRx(3,1),backend=FakeManilaV2(),initial_layout=[0,1,4])
   plt = fm.draw(style={'fontsize':15,'subfontsize': 10})
   plt.tight_layout()
   plt