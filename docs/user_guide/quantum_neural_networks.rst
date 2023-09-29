.. _quantum_neural_networks:

.. currentmodule:: squlearn.qnn

=======================
Quantum Neural Networks
=======================

Quantum Neural Networks (QNNs) extend the concept of artificial neural networks into the realm of
quantum computing. Typically, they are constructed by encoding input data into a quantum state
through a sequence of quantum gates. This quantum state is then manipulated using trainable
parameters and utilized to evaluate an expectation value of an observable that acts as the output
of the QNN. This output is then used to calculate a loss function, which is subsequently
minimized by a classical optimizer. The resultant QNN can then be employed to predict outcomes
for new input data.

In many cases, QNNs adhere to a layered design, akin to classical neural networks, as illustrated
in :numref:fig_qnn. However, it is essential to note that they do not adhere to the concept of
neurons as seen in classical neural networks. Therefore, the term "Quantum Neural Network" may
be somewhat misleading, as QNNs do not conform to the traditional neural network paradigm.
Nevertheless, their application domain closely resembles that of classical neural networks,
which explains the established nomenclature.

.. _fig_qnn:
.. figure:: ../_static/qnn/qnn.svg
    :alt: Quantum Neural Network (QNN)
    :width: 600
    :align: center

    Layered design of a QNN with alternating encoding (orange) and parameter (blue) layers.
    The QNN is trained in a hybrid quantum-classical scheme by optimizing the QNN's parameters
    :math:`{\theta}` for a given cost function :math:`L`.

In principle, the design of QNN architectures offers a high degree of freedom.
Nevertheless, most common designs follow a layered structure, where each layer comprises an
encoding layer denoted as :math:`U_i({x})` and a parameterized layer represented as
:math:`U_i({\theta})`. The encoding layers map the input data, :math:`{x}`, to a quantum state of
the qubits, while the parameterized layers are tailored to modify the mapped state.

The selection of the encoding method depends on the specific problem and the characteristics of
the input data, whereas parameterized layers are explicitly designed to alter the mapped state.
Furthermore, entanglement among the qubits is introduced, enabling the QNN to process information
in a more intricate and interconnected manner. Finally, we repeatedly measure the resulting state,
denoted as :math:`\Psi({x}, {\theta})`, to evaluate the QNN's output as the expectation value:

.. math::
    f({x}, {\theta}) = \langle\Psi({x}, {\theta}) \lvert\hat{C}({\theta})
    \rvert\Psi({x}, {\theta}) \rangle

Here, :math:`\hat{C}({\theta})` represents a operator, also called observable, for each output
of the QNN. While the observable can be freely selected, it often involves operators based on a
specific type of Pauli matrices, such as the Pauli Z matrix, to simplify the
evaluation of the expectation.

It's worth noting that both the embedding layers :math:`U_i({x})` and the observable
:math:`\hat{C}` may also contain additional trainable parameters.

To train Quantum Neural Networks (QNNs), a hybrid quantum-classical approach is employed.
The training process consists of two phases: quantum circuit evaluation and classical optimization
(as illustrated in :numref:`fig_qnn`).

In the quantum circuit evaluation phase, the QNN and its gradient with respect to the parameters
are assessed using a quantum computer or simulator. The gradient can be obtained using the
parameter-shift rule. Subsequently, in the classical optimization phase, an appropriate
classical optimization algorithm is employed to update the QNN's parameters.
This iterative process is repeated until the desired level of accuracy is attained.

Commonly used classical optimizers, such as SLSQP (for simulators) or stochastic gradient
descent, like Adam, are applied in the classical optimization stage of QNN training.
They adjust the QNN's parameters to minimize a predefined cost function, denoted as :math:`L`:

.. math::
    \min_{\theta} L(f, {x}, {\theta})

The specific form of the cost function depends on the problem that the QNN is designed to solve.
For instance, in a regression problem, the cost function is often defined as the mean squared
error between the QNN's output and the target value.

High-level methods for QNNs
====================================

In this section, we will demonstrate how to construct a simple QNN using sQUlearn.
A QNN is constructed from an encoding circuit, i.e. a parameterized quantum circuit, and a cost
operator.
In sQUlearn there are specific classes for the encoding circuits, :class:`EncodingCircuit`,
and the cost operators, :class:`CostOperator`, which we will use in the following example.

In the example we construct a encoding circuit based on the Chebyshev input encoding:

.. code-block:: python

    from squlearn.feature_map import ChebPQC

    pqc = ChebPQC(num_qubits = 4, num_features = 2, num_layers = 2)
    pqc.draw("mpl")

.. plot::

    from squlearn.feature_map import ChebPQC
    pqc = ChebPQC(4, 2, 2)
    pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
    plt.tight_layout()

There are several other encoding circuits available in sQUlearn, which can be found in the
user guide on :ref:`quantum_feature_maps`.

Additionally we have to define a observable to compute the oupout of the QNN. In this example
we use a summation over a Pauli Z observable for each qubit and a constant offset:

.. code-block:: python

    from squlearn.expectation_operator import SummedPaulis

    op = SummedPaulis(num_qubits=4)

.. plot::
    from squlearn.expectation_operator import SummedPaulis
    op = SummedPaulis(num_qubits=4)
    print(op)


Other expectation operators can be found in the user guide on :ref:`operators`.

Now we can construct a QNN from the encoding circuit and the cost operator.
sQUlearn offers two easy-to-use implementation of QNNs, either for regression or classification:

.. autosummary::
   :nosignatures:

   QNNClassifier
   QNNRegressor

We refer to the documentations and examples of the respective classes for in-depth information.

In the following example we will use a :class:`QNNRegressor`, the encoding circuit and
expectation operator as defined above, as well as a mean squared error loss function and the
Adam optimizer for optimization.

.. code-block:: python

    from squlearn.expectation_operator import SummedPaulis
    from squlearn.feature_map import ChebPQC
    from squlearn.qnn import QNNRegressor, SquaredLoss
    from squlearn.optimizers import Adam
    from squlearn import Executor

    op = SummedPaulis(num_qubits = 4)
    pqc = ChebPQC(num_qubits = 4, num_features = 2, num_layers = 2)
    qnn = QNNRegressor(pqc, op, Executor("statevector_simulator"), SquaredLoss(), Adam())

The QNN can be trained utilizing the :meth:`fit <squlearn.qnn.QNNRegressor.fit>` method:

.. code-block:: python

    import numpy as np
    # Data that is inputted to the QNN
    x_train = np.arange(-0.5, 0.6, 0.1)
    # Data that is fitted by the QNN
    y_train = np.square(x_train)

    qnn.fit(x_train, y_train)

The inference of the QNN can be calculated using the :meth:`predict <squlearn.qnn.QNNRegressor.predict>` method:

.. code-block:: python

    x_test = np.arange(-0.5, 0.5, 0.01)
    y_pred = qnn.predict(x_test)


Optimization
============

To train a QNN's parameters, sQUlearn offers a lot of possibilities for modification. In this
section we will show, how to use :class:`SLSQP <squlearn.optimizers.optimizers_wrapper.SLSQP>`,
as an example for a wrapped scipy optimizer, and :class:`Adam <squlearn.optimizers.adam.Adam>`
with mini-batch gradient descent to optimize the loss function.

SLSQP
-----

sQUlearn offers wrapper functions, :class:`SLSQP <squlearn.optimizers.optimizers_wrapper.SLSQP>`
and :class:`LBFGSB <squlearn.optimizers.optimizers_wrapper.LBFGSB>`, for scipy's SLSQP and
L-BFGS-B implementations as well as the wrapper function
:class:`SPSA <squlearn.optimizers.optimizers_wrapper.SPSA>` for Qiskit's SPSA implementation.
We show how to import and use :class:`SLSQP <squlearn.optimizers.optimizers_wrapper.SLSQP>`
in the following code block.

.. code-block:: python

    from squlearn.optimizers import SLSQP

    ...

    slsqp = SLSQP(options=options_dict)

    ...

    reg = QNNRegressor(
        ...
        optimizer=slsqp,
        ...
    )

With this configuration, :class:`QNNRegressor` will use scipy's
:func:`minimize <scipy.optimize.minimize>` function with ``method="SLSQP"``.
The wrapper Class :class:`SLSQP <squlearn.optimizers.optimizers_wrapper.SLSQP>`
allows to specify hyper parameters in a :class:`dict` that get passed on to the function.

Mini-Batch gradient descent with Adam
-------------------------------------

sQUlearn's QNN classes, :class:`QNNRegressor` and :class:`QNNClassifier`, also offer the
possibility to use mini-batch gradient descent with Adam to optimize the model. This allows for
training on bigger data sets. Therefore we import and use the
:class:`Adam <squlearn.optimizers.adam.Adam>` optimizer as demonstrated in the following
code block.

.. code-block:: python

    from squlearn.optimizers import Adam

    ...

    adam = Adam(options=options_dict)

    ...

    reg = QNNRegressor(
        ...
        optimizer=adam,
        ...
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        ...
    )

Using the :class:`Adam <squlearn.optimizers.adam.Adam>` optimizer allows us to specify further
hyper parameters such as ``batch_size``, ``epochs`` and ``shuffle``. ``batch_size`` and ``epochs``
are positive numbers of type :class:`int` and ``shuffle`` is a :class:`bool` which specifies,
whether data points are shuffled before each epoch.

Variance reduction
==================

When evaluating a pretrained QNN on Qiskit's :class:`QasmSimulator <qiskit_aer.QasmSimulator>` or
on real hardware, the model will be subject to randomness due to the inherent nature of
quantum mechanics. The overall performance of the model thus depends on its variance, which can
be calculated as

.. math::

    \sigma_f^2 = \langle\Psi\lvert\hat{C}^2\rvert\Psi\rangle -
    \langle\Psi\lvert\hat{C}\rvert\Psi\rangle^2 \text{.}

:numref:`fig_qnn_output_high_var` shows the output of a :class:`QNNRegressor` fit to a logarithm
with :class:`SquaredLoss <squlearn.qnn.loss.SquaredLoss>` evaluated on Qiskit's
:class:`QasmSimulator <qiskit_aer.QasmSimulator>`. The model is subject to high variance.

.. _fig_qnn_output_high_var:
.. figure:: ../_static/qnn/qnn_output_high_var.svg
    :alt: QNN Output with high variance
    :width: 600
    :align: center

    Logarithm and output of :class:`QNNRegressor` :math:`f(\theta, x)` evaluated on Qiskit's
    :class:`QasmSimulator <qiskit_aer.QasmSimulator>`. The QNN output has a high variance.

We can mitigate this problem by adding the models variance to the loss function
:math:`L_\text{fit}` and thus regularizing for variance. We do this by setting the `variance`
keyword in the initialization of the :class:`QNNRegressor` (or :class:`QNNClassifier`) with a
hyper-parameter :math:`\alpha`.

.. code-block:: python

    reg = QNNRegressor(
        ...
        variance = alpha,
        ...
    )

The new total loss function reads as

.. math::

    L_\text{total} = L_\text{fit} +
    \alpha \cdot \sum_k \lVert \sigma_f^2 ( x_i )\rVert \text{,}

where :math:`\sigma_f^2( x_i )` is the variance of the QNN on the training data
:math:`\{x_i\}`. This approach also allows us to reuse the circuits and function evaluations
needed for calculating :math:`L_\text{fit}`.

The regularization factor :math:`\alpha` controls the influence of the variance regularization on
the total loss. It can be either set to a constant :class:`float` or a :class:`Callable` that
takes the keyword argument ``iteration`` to dynamically adjust the factor. Values between
:math:`10^{-2}` and :math:`10^{-4}` have shown to yield satisfying results. `[1]`_

Evaluation on Qiskit's :class:`QasmSimulator <qiskit_aer.QasmSimulator>` now yields less variance
in the model, as depicted in :numref:`fig_qnn_output_low_var`.

.. _fig_qnn_output_low_var:
.. figure:: ../_static/qnn/qnn_output_low_var.svg
    :alt: QNN Output with low variance
    :width: 600
    :align: center

    Logarithm and output of :class:`QNNRegressor` :math:`f(\theta, x)`, trained with variance
    regularization, evaluated on Qiskit's :class:`QasmSimulator <qiskit_aer.QasmSimulator>`.
    The QNN output has a low variance.

.. rubric:: References

_`[1]` D. A. Kreplin and M. Roth "Reduction of finite sampling noise in quantum neural networks".
`arXiv:2306.01639 <https://arxiv.org/abs/2306.01639>`_ (2023).
