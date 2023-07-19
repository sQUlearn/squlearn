################################
Operators for expectation values
################################

.. currentmodule:: squlearn.expectation_operator

Operators play a crucial role in computing the expectation value in conjunction with a
wavefunction. Currently, in sQUlearn, only operators constructed from the Pauli group
:math:`\{\hat{X},\hat{Y},\hat{Z},\hat{I}\}` are supported. In the context of sQUlearn,
expectation operators are mandatory inputs for the Quantum Neural Network (QNN) and can be
employed in the Projected Quantum Kernel program.
All operators follow the Base Class :class:`ExpectationOperatorBase`.

The following functions and operators are accessible via :class:`squlearn.expectation_operator`.

Implemented expectation operators.
----------------------------------

The following operators are available and implemented as standalone classes:

.. autosummary::
    :nosignatures:

    SinglePauli
    SummedPaulis
    SingleProbability
    SummedProbabilities
    IsingHamiltonian
    CustomExpectationOperator

**Example: Summed Pauli Operator**

.. code-block:: python

    from squlearn.expectation_operator import SummedPaulis

    op = SummedPaulis(num_qubits=2)
    print(op)


**Example: Custom operator**

Operators can be added by ``+`` or multiplied by ``*`` with each other.
Together with the class :class:`CustomExpectationOperator`, in which single parameterized operators
can be build from strings, your can create more complex operators.

.. code-block:: python

    from squlearn.expectation_operator import CustomExpectationOperator

    op1 = CustomExpectationOperator(num_qubits=2, operator_string="IX",parameterized=True)
    op2 = CustomExpectationOperator(num_qubits=2, operator_string="ZY",parameterized=True)
    total_op = op1 + op2
    print(total_op)

Note that in Qiskit, the qubits are counted from the right to the left as in the computational
basis!


**Example: Use the mapping from the transpiled feature map**

When running on a backend, the number of physical qubits may change
from the number of qubits the definition of the expectation operator.
to solve this issue, it is possible to provide a map from the logical qubits to the
physical qubits via :meth:`set_qubit_map`.
The map can be for example obtained in the transpiled feature map.

.. code-block:: python

   from squlearn.feature_map import ChebRx,TranspiledFeatureMap
   from squlearn.expectation_operator import SummedPaulis
   from qiskit.providers.fake_provider import FakeManilaV2
   fm = TranspiledFeatureMap(ChebRx(3,1),backend=FakeManilaV2(),initial_layout=[0,1,4])
   op = SummedPaulis(num_qubits=3, op_str="Z")
   op.set_map(fm.qubit_map, fm.num_all_qubits)
   print(op)


Obtained derivatives of the expectation values
----------------------------------------------

In sQUlearn it is also possible to evaluate the derivatives of expectation operators
as for example needed during the training of the QNN.
This is possible with the class :class:`ExpectationOperatorDerivatives`.
The derivatives are calculated with respect to the parameters of the expectation operator.

**Example: first-order derivative of the Ising Hamiltonian**

.. code-block:: python

    from squlearn.expectation_operator import IsingHamiltonian,ExpectationOperatorDerivatives
    op = IsingHamiltonian(num_qubits=3)
    print(ExpectationOperatorDerivatives(op).get_derivative("dop"))

To calculate the variance of an operator, the squared operator can be used.

**Example: Squared summed Pauli Operator**

.. code-block:: python

    from squlearn.expectation_operator import SummedPaulis,ExpectationOperatorDerivatives
    op = SummedPaulis(num_qubits=3)
    print(ExpectationOperatorDerivatives(op).get_operator_squared())

In principle, arbitrary derivatives can be computed by supplying a tuple, but often, the higher-order derivatives
are zero. To achieve this, the function :meth:`ExpectationOperatorDerivatives.get_derivative` can be used with a tuple
of parameters :meth:`ExpectationOperatorDerivatives.parameter_vector`.

**Example: higher-order derivative of the cubed SummedPaulis operator**

.. code-block:: python

   from squlearn.expectation_operator import SummedPaulis,ExpectationOperatorDerivatives

   # Build cubed SummedPaulis operator
   op = SummedPaulis(num_qubits=2)
   op3 = op*op*op
   print(op3)

   # Get the Hessian from a tuple
   deriv = ExpectationOperatorDerivatives(op3)
   print(deriv.get_derivative((deriv.parameter_vector[0],deriv.parameter_vector[0])))