.. _user_guide_observables:

##################################
Observables for expectation values
##################################

.. currentmodule:: squlearn.observables

Observables play a crucial role in computing the expectation value in conjunction with a
wave function. Currently, observables constructed purely from the Pauli group
:math:`\{\hat{X},\hat{Y},\hat{Z},\hat{I}\}` are supported. In the context of sQUlearn,
observables are mandatory inputs for Quantum Neural Networks (QNNs) and can be
employed in the Projected Quantum Kernel program.
All operators follow the Base Class :class:`ObservableBase`. sQulearn features several
predefined observables, but it is also possible to simply construct custom observables.

Implemented observables.
----------------------------------

The following predefined observables are available in the module :class:`squlearn.observables`:

.. autosummary::
    :nosignatures:

    SinglePauli
    SummedPaulis
    SingleProbability
    SummedProbabilities
    IsingHamiltonian
    CustomObservable

The observables are simply constructed by initializing the associated class.

**Example: Summed Pauli observable**

.. jupyter-execute::

    from squlearn.observables import SummedPaulis

    op = SummedPaulis(num_qubits=2)
    print(op)

Custom observables
--------------------------


sQUlearn features several options to construct custom observables.
With the class :class:`CustomObservable`, it is possible to construct observables from a string
containing the letters of the Pauli matrices (``"X"``, ``"Y"``, ``"Z"``, ``"I"``).
The resulting observable can be multiplied by a parameter by setting ``parameterized=True``.
Furthermore, observables can be added by ``+`` or multiplied by ``*`` with each other.
This allows the creation of arbitrary observables.

**Example: Custom observable**

.. jupyter-execute::

    from squlearn.observables import CustomObservable

    ob1 = CustomObservable(num_qubits=2, operator_string="IX",parameterized=True)
    ob2 = CustomObservable(num_qubits=2, operator_string="ZY",parameterized=True)
    added_ob = ob1 + ob2
    print("Added observable:\n",added_ob,"\n\n")
    squared_ob = added_ob*added_ob
    print("Squared observable:\n",squared_ob)

**Example: More complex custom observable**

It is also possible to construct more complex observables by supplying a list of strings
containing the Pauli matrices. The observable is then constructed by adding the single observables
together.

.. jupyter-execute::

    from squlearn.observables import CustomObservable

    # It is also possible to add trainable parameters:
    ob = CustomObservable(num_qubits=4, operator_string=["ZIZZ", "XIXI"], parameterized=True)
    print("Custom observable with multiple operators:\n",ob)


Note that in Qiskit, the qubits are counted from the right to the left as in the computational
basis!

Mapping observables to real qubits
------------------------------------


When running on a backend, the number of physical qubits may change
from the number of qubits the definition of the observable.
If it is necessary, it is possible to adjust the observable to the physical qubits.
This is achieved by providing a map from the qubits of the observable to the
physical qubits utilized for example in the feature map via the function :meth:`set_qubit_map`.
The map can be for example obtained in the transpiled encoding circuit.

**Example: Use the mapping from the transpiled encoding circuit**

.. jupyter-execute::

   from squlearn.encoding_circuit import ChebyshevRx,TranspiledEncodingCircuit
   from squlearn.observables import SummedPaulis
   from qiskit_ibm_runtime.fake_provider import FakeManilaV2
   fm = TranspiledEncodingCircuit(ChebyshevRx(3,1),backend=FakeManilaV2(),initial_layout=[0,1,4])
   ob = SummedPaulis(num_qubits=3, op_str="Z")
   print("Observable before mapping:\n",ob,"\n\n")
   ob.set_map(fm.qubit_map, fm.num_physical_qubits)
   print("Observable after mapping:\n",ob)


Derivatives of the observable
----------------------------------------------

.. currentmodule:: squlearn.observables.observable_derivatives

In sQUlearn it is also possible to calculate derivatives of observables
as for example needed during the training of the QNN.
This is possible with the class :class:`ObservableDerivatives`.
The derivatives are calculated with respect to the parameters of the observable.


**Example: first-order derivatives of the Ising Hamiltonian**

.. jupyter-execute::

    from squlearn.observables import IsingHamiltonian
    from squlearn.observables.observable_derivatives import ObservableDerivatives
    ob = IsingHamiltonian(num_qubits=3)
    print("Observable:\n", ob,"\n\n")
    print("Gradient of the observable:\n", ObservableDerivatives(ob).get_derivative("dop"))


**Example: higher-order derivative of the cubed SummedPaulis observable**

Furthermore, arbitrary derivatives can be computed by supplying a tuple, although the
higher-order derivatives are zero for linear parameters.
To achieve this, the function :meth:`ObservableDerivatives.get_derivative` can be used with a tuple
of parameters :meth:`ObservableDerivatives.parameter_vector`.

.. jupyter-execute::

   from squlearn.observables import SummedPaulis
   from squlearn.observables.observable_derivatives import ObservableDerivatives

   # Build cubed SummedPaulis observable
   op = SummedPaulis(num_qubits=2)
   op3 = op*op*op
   print("Cubed operator:\n",op3)

   # Get the Hessian from a tuple
   deriv = ObservableDerivatives(op3)
   print("Second-order derivative w.r.t. p[0]:\n",
            deriv.get_derivative((deriv.parameter_vector[0],deriv.parameter_vector[0])))


**Example: Squared summed Pauli Observable**

It is also possible to calculate the squared observable, which is needed for the
calculation of the variance of the observable.

.. jupyter-execute::

    from squlearn.observables import SummedPaulis
    from squlearn.observables.observable_derivatives import ObservableDerivatives
    op = SummedPaulis(num_qubits=3)
    print(ObservableDerivatives(op).get_operator_squared())


