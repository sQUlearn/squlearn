.. _quantum_encoding_circuits:

.. currentmodule:: squlearn.encoding_circuit

##########################
Quantum Encoding Circuits
##########################

Quantum encoding circuits are used to embed classical data :math:`x` into a quantum state and manipulate
the quantum state via trainable parameters :math:`p`.
They are a key component of many quantum machine learning algorithms, and the design of a good
quantum encoding circuit is crucial for the performance of the algorithm.
In sQUlearn, encoding circuits are an obligatory input in the Quantum Neural Network (QNN) or Quantum
Kernel programs.
sQUlearn offers a wide range of pre-implemented quantum encoding circuits,
which can be combined to create more sophisticated encoding circuits.
Furthermore, it is possible to create custom encoding circuits that follow a layered approach, in which
each gate is applied to all qubits.
The package facilitate a fully automated pruning algorithm to remove redundant parameters and
enables the automatic differentiation of arbitrary derivative.

The following functions and classes are are accessible via :class:`squlearn.encoding_circuit`.

Implemented Quantum Encoding Circuits
--------------------------------------

There are several Quantum Encoding Circuits implemented in sQUlearn:

.. autosummary::
   :nosignatures:

   YZ_CX_EncodingCircuit
   HighDimEncodingCircuit
   HubregtsenEncodingCircuit
   ChebyshevTower
   ChebyshevPQC
   MultiControlEncodingCircuit
   ChebyshevRx
   ParamZFeatureMap
   QiskitEncodingCircuit

Feel free to contribute to sQUlearn by adding your own encoding circuits in a Pull Request.


**Example: Create a Hubregtsen encoding circuit**

.. jupyter-execute::

   from squlearn.encoding_circuit import HubregtsenEncodingCircuit
   pqc = HubregtsenEncodingCircuit(num_qubits=4, num_features=2, num_layers=2)
   pqc.draw(output="mpl")


Combining Quantum Encoding Circuits
--------------------------------------
In sQUlearn, quantum encoding circuits can be combined to create more sophisticated encoding circuits by
utilizing the ``+`` operation between two encoding circuits. However, it is important to note that
the number of qubits in both encoding circuits must match for successful combination.

When combining encoding circuits, the resulting feature dimension is determined by taking the maximum
value from the two feature dimensions. The parameters of the individual encoding circuits are
concatenated. Consequently, the total number of parameters in the combined encoding circuit is
equal to the sum of the parameters in the two original encoding circuits.

**Example: combine two quantum encoding circuits**

.. jupyter-execute::

   from squlearn.encoding_circuit import HubregtsenEncodingCircuit, ChebyshevPQC
   fm1 = HubregtsenEncodingCircuit(num_qubits=4, num_features=2, num_layers=1, closed=False)
   fm2 = ChebyshevPQC(num_qubits=4, num_features=3, num_layers=1)
   # Combining both encoding circuits
   fm3 = fm1 + fm2
   fm3.draw(output="mpl")


Wrapping Qiskit Encoding Circuits
--------------------------------------

It is also possible to utilize the wrapper :class:`QiskitEncodingCircuit` to build Encoding Circuits from the
`Qiskit circuit library <https://qiskit.org/documentation/apidoc/circuit_library.html>`_.

.. jupyter-execute::

   from squlearn.encoding_circuit import QiskitEncodingCircuit
   from qiskit.circuit.library import TwoLocal
   local = TwoLocal(3, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
   QiskitEncodingCircuit(local).draw(output="mpl")


Create your custom Encoding Circuit via :class:`LayeredEncodingCircuit`
------------------------------------------------------------------------

sQUlearn offers a user-friendly solution for creating custom layered encoding circuits effortlessly.
Layered encoding circuits involve the application of gates to all qubits, ensuring a comprehensive
approach. This method allows for the creation of encoding circuits in a structured manner,
regardless of the number of qubits involved.
Two-qubit gates are applied either in a nearest neighbor fashion or by entangling all qubits.
You can construct the layered encoding circuit using either a Qiskit Quantum circuit-inspired approach
or by providing a string using the
:meth:`LayeredEncodingCircuit.from_string() <squlearn.encoding_circuit.LayeredEncodingCircuit.from_string>`
method. For detailed instructions on the string format, please refer to the documentation of
the :class:`LayeredEncodingCircuit` class.

**Example: Create your custom layered encoding circuit**

.. jupyter-execute::

   from squlearn.encoding_circuit import LayeredEncodingCircuit
   from squlearn.encoding_circuit.layered_encoding_circuit import Layer
   encoding_circuit = LayeredEncodingCircuit(num_qubits=4,num_features=2)
   encoding_circuit.H()
   layer = Layer(encoding_circuit)
   layer.Rz("x")
   layer.Ry("p")
   layer.cx_entangling("NN")
   encoding_circuit.add_layer(layer,num_layers=3)
   encoding_circuit.draw(output="mpl")


**Example: Create your custom layered encoding circuit from a string**

.. jupyter-execute::

   from squlearn.encoding_circuit import LayeredEncodingCircuit
   encoding_circuit = LayeredEncodingCircuit.from_string(
      "Ry(p)-3[Rx(p,x;=y*np.arccos(x),{y,x})-crz(p)]-Ry(p)", num_qubits=4, num_features=1, num_layers=2
   )
   encoding_circuit.draw(output="mpl")


Pruning of Quantum Encoding Circuits
--------------------------------------

It is also possible to remove parameterized gates from a quantum encoding circuit by using the
:class:`PrunedEncodingCircuit` class.
This class accepts a quantum encoding circuit as input and removes the parameterized gates
from the encoding circuit for the parameters which indices are specified in the supplied list.
The pruned encoding circuit automatically adjusts the number of parameters and features.

Furthermore it is possible to determine the redundant parameters in encoding circuit automatically.
The algorithm is based on https://doi.org/10.1103/PRXQuantum.2.040309 and is based on evaluating
the Quantum Fisher Information Matrix (QFIM) of the encoding circuit.

sQUlearn features a fully automated pruning algorithm which can be used by calling the routine
:meth:`automated_pruning` that returns a pruned encoding circuit without the redundant parameters.

**Example: Pruning a encoding circuit with redundant parameters**

.. jupyter-execute::

   from squlearn.encoding_circuit import LayeredEncodingCircuit, automated_pruning
   from squlearn.util import Executor
   encoding_circuit = LayeredEncodingCircuit.from_string("Rz(p)-Ry(p)-Z-Ry(p)-Rz(p)", num_qubits=2, num_features=0)
   pruned_encoding_circuit = automated_pruning(encoding_circuit, Executor())
   pruned_encoding_circuit.draw(output="mpl")


Different Quantum Encoding Circuits via :class:`EncodingCircuitDerivatives`
----------------------------------------------------------------------------

The calculation of derivatives for quantum encoding circuits is often essential in training a
Quantum Machine Learning model. In sQUlearn, we offer a straightforward approach to compute
these derivatives using the :class:`EncodingCircuitDerivatives` class.
This class accepts an existing quantum encoding circuit as input and generates derivatives of the
encoding circuit with respect to its parameters or features. The derivative circuits are generated
by leveraging the parameter-shift rule and are cached for future use.
Use the function :meth:`get_derivative() <squlearn.encoding_circuit.EncodingCircuitDerivatives.get_derivative>`
to obtain the derivative. There are several options to specify the derivative you want to obtain:

1. Provide a string that specifies the derivative you want to obtain. A list of the available
   strings can be found in the documentation of the :class:`EncodingCircuitDerivatives` class.
2. Provide a tuple containing the ParameterVector or an element of the ParameterVector to
   obtain higher order derivatives. The derivatives are applied successively following the order in
   the tuple.
3. Provide a list of ParameterVector elements to obtain the derivatives of the specified elements.
   This can also be placed in the tuple.

The derivatives are stored in sQUlearn's proprietary OpTree structure, which
is utilized for the arithmetic operations of the derivatives.

**Example: Obtain the derivative of a Hubregtsen encoding circuit**

.. jupyter-execute::

   from squlearn.encoding_circuit import HubregtsenEncodingCircuit, EncodingCircuitDerivatives
   fm = HubregtsenEncodingCircuit(num_qubits=2, num_features=2, num_layers=2)
   fm_deriv = EncodingCircuitDerivatives(fm)
   # From String (gradient of the parameter vector)
   grad_from_string = fm_deriv.get_derivative("dp")
   # From Tuple (second order derivative of the parameter vector; equal to the Hessian)
   grad_from_tuple = fm_deriv.get_derivative((fm_deriv.parameter_vector,fm_deriv.parameter_vector))
   # From List (only partial derivatives of the first two parameters)
   grad_from_List = fm_deriv.get_derivative(([fm_deriv.parameter_vector[0],
                                               fm_deriv.parameter_vector[1]],
                                             ))

Transpile Quantum Encoding Circuits via :class:`TranspiledEncodingCircuit`
---------------------------------------------------------------------------

To transpile a quantum encoding circuit, you can leverage the functionality provided by the
:class:`TranspiledEncodingCircuit` class. By utilizing this class, you can input an existing
quantum encoding circuit and have its circuit transpiled according to the specified backend and
transpiler settings, which are the same settings used in Qiskit.
The transpiled encoding circuit is internally employed in the QNN program and projected kernels,
where it is employed internally.

**Example: Transpile a existing Encoding Circuit to a fake backend**

.. jupyter-execute::

   from squlearn.encoding_circuit import TranspiledEncodingCircuit,ChebyshevRx
   from qiskit_ibm_runtime.fake_provider import FakeManilaV2

   fm = TranspiledEncodingCircuit(ChebyshevRx(3,1),backend=FakeManilaV2(),initial_layout=[0,1,4])
   fm.draw(output="mpl")

