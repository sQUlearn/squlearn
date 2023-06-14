################################
Operators for expectation values
################################

The operators from :class:`~squlearn.expectation_operator` can be used to calculate the expectation value of
a given operator. At the current state of sQUlearn, only operators from the Pauli group are supported.
The following operators are available and implemented as standalone classes:


 * :class:`~squlearn.expectation_operator.SinglePauli`: This operator calculates the expectation value of a single Pauli operator.
 * :class:`~squlearn.expectation_operator.SummedPaulis`: This operator calculates the expectation value of a sum of Pauli operators.
 * :class:`~squlearn.expectation_operator.SingleProbability`: This operator returns the of the probability of being in the one or zero state.
 * :class:`~squlearn.expectation_operator.SummedProbabilities`: This operator returns the sum of the probabilities of being in the one or zero state.
 * :class:`~squlearn.expectation_operator.IsingHamiltonian`: This operator provides a structure for several Ising Hamiltonians.
 * :class:`~squlearn.expectation_operator.CustomExpectationOperator`: This class can be used to implement custom expectation operators from strings.

All operators follow the structure of the base class :class:`~squlearn.expectation_operator.ExpectationOperatorBase` class:

.. autoclass:: squlearn.expectation_operator.ExpectationOperatorBase
  :members:

Derivatives of the expectation value with respect to the parameters can be calculated using the :class:`~squlearn.expectation_operator.ExpectationOperatorDerivatives` class.


Example
-------

The following example shows the use single Pauli operator:

.. code-block:: python

    from squlearn.expectation_operator import SummedPaulis
    from qiskit.circuit import ParameterVector

    op = SummedPaulis(num_qubits=4, op_str="Z")
    param = ParameterVector("p", op.num_parameters)
    print(op.get_pauli(param))

More examples of the different operators can be found in the tutorial notebook ``./examples/qnn/expectation_operator.ipynb``.

In sQUlearn, the operators can be used in the following parts of the program:

 * :class:`~squlearn.qnn`: The expectation value of the operator is used to calculate the output of the QNN.
 * :class:`~squlearn.kernel.matrix.pqk`: The expectation operator can be used in the projected quantum kernel.

