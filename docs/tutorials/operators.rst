################################
Operators for expectation values
################################

.. currentmodule:: squlearn.expectation_operator

The operators from :class:`~squlearn.expectation_operator` can be used to calculate the expectation value of
a given operator. At the current state of sQUlearn, only operators from the Pauli group are supported.
The following operators are available and implemented as standalone classes:

.. autosummary::
    :nosignatures:

    SinglePauli
    SummedPaulis
    SingleProbability
    SummedProbabilities
    IsingHamiltonian
    CustomExpectationOperator


All operators follow the structure of the base class :class:`expectation_operator_base.ExpectationOperatorBase` class:

.. autoclass:: squlearn.expectation_operator.expectation_operator_base.ExpectationOperatorBase
  :members:

Derivatives of the expectation value with respect to the parameters can be calculated using the :class:`~squlearn.expectation_operator.expectaion_operator_derivatives.ExpectationOperatorDerivatives` class.


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

