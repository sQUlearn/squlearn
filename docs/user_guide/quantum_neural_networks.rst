.. _quantum_neural_networks:

.. currentmodule:: squlearn.qnn

=======================
Quantum Neural Networks
=======================

Quantum Neural Networks (QNNs) are a type of machine learning model that combine the principles of quantum computing and neural networks. They are usually built using a alternating layered structure, as depicted in :numref:`fig_qnn`, which is reminiscent of a classical deep neural network and thus gives them their name. The layers alternate between embedding- and parameterized layers.

.. _fig_qnn:
.. figure:: ../_static/qnn/qnn.svg
    :alt: Quantum Neural Network (QNN)
    :width: 600
    :align: center

    Layered design of a QNN with alternating embedding (orange) and entangling (blue) layers. The QNN is trained in a hybrid quantum-classical scheme by optimizing the QNN's parameters :math:`\vec{\theta}` for a given cost function :math:`L`.

Embedding layers are responsible for mapping the input data to the quantum state of the qubits. They encode the input features into the amplitudes or other properties of the quantum state. The choice of embedding depends on the specific problem and the nature of the input data. Parameterized layers are designed to alter the mapped state and introduce entanglement among the qubits, enabling the QNN to process information in a more complex and interconnected manner. They typically consist of quantum gates that, as the name states, create entanglement between different qubits. Examples of commonly used entangling gates include the Controlled-Z gate, Controlled-X gate, or other two-qubit gates. Finally we measure the resulting state :math:`\Psi(\vec{x}, \vec{\theta})` repeatedly to gain the output of the QNN as the expectation value

.. math::
    f\left(\vec{x}, \vec{\theta}\right) = \left\langle\Psi\left(\vec{x}, \vec{\theta}\right)\left\lvert\hat{C}\left(\vec{\theta}\right)\right\rvert\Psi\left(\vec{x}, \vec{\theta}\right)\right\rangle

for some cost operator :math:`\hat{C}(\vec{\theta})`. Per desired output of the QNN, we need to specify one cost operator.

Both, the embedding layers :math:`U_i(\vec{\theta}, \vec{x}), i=1,\dots,L` and the entangling layers :math:`U_i(\vec{\theta_i}), i=1,\dots,L`, as well as the cost operators :math:`\hat{C}`, can contain trainable parameters :math:`\vec{\theta}`. 

To train QNNs, a hybrid quantum-classical scheme is used. The training process involves two steps: quantum circuit evaluation and classical optimization (compare :numref:`fig_qnn`). In the quantum circuit evaluation step, the QNN is evaluated on a quantum computer or simulator to generate the output. The classical optimization step involves using a fitting classical optimization algorithm to update the parameters of the QNN based on the difference between the expected output and the actual output obtained from the quantum circuit execution. This process is repeated iteratively until the desired level of accuracy is achieved. 

Classical optimizers, such as SLSQP or stochastic gradient descent, are commonly used in the classical optimization step of training QNNs. They update the parameters of the QNN to minimize the program

.. math::
    \min_\vec{\theta} L\left(f, \vec{x}, \vec{\theta}\right)

according to a defined cost function :math:`L`. The cost function represents the discrepancy between the expected output and the actual output obtained from the QNN evaluation.

High-Level methods for QNNs
===========================

sQUlearn offers easy-to-use implementation of QNNs for regression and classification.

.. autosummary::
   :nosignatures:

   QNNClassifier
   QNNRegressor

We refer to the documentations and examples of the respective classes for in-depth information and user guidelines.

Optimization
============

To train a QNN's parameters, sQUlearn offers a lot of possibilities for modification. In this section we will show, how to use :class:`SLSQP <squlearn.optimizers.optimizers_wrapper.SLSQP>`, as an example for a wrapped scipy optimizer, and :class:`Adam <squlearn.optimizers.adam.Adam>` with mini-batch gradient descent to optimize the loss function.

SLSQP
-----

sQUlearn offers wrapper functions, :class:`SLSQP <squlearn.optimizers.optimizers_wrapper.SLSQP>` and :class:`LBFGSB <squlearn.optimizers.optimizers_wrapper.LBFGSB>`, for scipy's SLSQP and L-BFGS-B implementations as well as the wrapper function :class:`SPSA <squlearn.optimizers.optimizers_wrapper.SPSA>` for Qiskit's SPSA implementation. We show how to import and use :class:`SLSQP <squlearn.optimizers.optimizers_wrapper.SLSQP>` in the following code block.

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

With this configuration, :class:`QNNRegressor` will use scipy's :func:`minimize <scipy.optimize.minimize>` function with ``method="SLSQP"``. The wrapper Class :class:`SLSQP <squlearn.optimizers.optimizers_wrapper.SLSQP>` allows to specify hyper parameters in a :class:`dict` that get passed on to the function.

Mini-Batch gradient descent with Adam
-------------------------------------

sQUlearn's QNN classes, :class:`QNNRegressor` and :class:`QNNClassifier`, also offer the possibility to use mini-batch gradient descent with Adam to optimize the model. This allows for training on bigger data sets. Therefore we import and use the :class:`Adam <squlearn.optimizers.adam.Adam>` optimizer as demonstrated in the following code block.

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

Using the :class:`Adam <squlearn.optimizers.adam.Adam>` optimizer allows us to specify further hyper parameters such as ``batch_size``, ``epochs`` and ``shuffle``. ``batch_size`` and ``epochs`` are positive numbers of type :class:`int` and ``shuffle`` is a :class:`bool` which specifies, whether data points are shuffled before each epoch.

Variance reduction
==================

When evaluating a pretrained QNN on Qiskit's :class:`QasmSimulator <qiskit_aer.QasmSimulator>` or on real hardware, the model will be subject to randomness due to the inherent nature of quantum mechanics. The overall performance of the model thus depends on its variance, which can be calculated as

.. math::

    \sigma_f^2 = \left\langle\Psi\left\lvert\hat{C}^2\right\rvert\Psi\right\rangle - \left\langle\Psi\left\lvert\hat{C}\right\rvert\Psi\right\rangle^2 \text{.}

:numref:`fig_qnn_output_high_var` shows the output of a :class:`QNNRegressor` fit to a logarithm with :class:`SquaredLoss <squlearn.qnn.loss.SquaredLoss>` evaluated on Qiskit's :class:`QasmSimulator <qiskit_aer.QasmSimulator>`. The model is subject to high variance.

.. _fig_qnn_output_high_var:
.. figure:: ../_static/qnn/qnn_output_high_var.svg
    :alt: QNN Output with high variance
    :width: 600
    :align: center

    Logarithm and output of :class:`QNNRegressor` :math:`f(\theta, x)` evaluated on Qiskit's :class:`QasmSimulator <qiskit_aer.QasmSimulator>`. The QNN output has a high variance.

We can mitigate this problem by adding the models variance to the loss function :math:`L_\text{fit}` and thus regularizing for variance. We do this by setting the `variance` keyword in the initialization of the :class:`QNNRegressor` (or :class:`QNNClassifier`) with a hyper-parameter :math:`\alpha`.

.. code-block:: python

    reg = QNNRegressor(
        ...
        variance = alpha,
        ...
    )

The new total loss function reads as

.. math::

    L_\text{total} = L_\text{fit} + \alpha \cdot \sum_k \lVert \sigma_f^2 \left( x_i \right)\rVert \text{,}

where :math:`\sigma_f^2( x_i )` is the variance of the QNN on the training data :math:`\{x_i\}`. This approach also allows us to reuse the circuits and function evaluations needed for calculating :math:`L_\text{fit}`.

The regularization factor :math:`\alpha` controls the influence of the variance regularization on the total loss. It can be either set to a constant :class:`float` or a :class:`Callable` that takes the keyword argument ``iteration`` to dynamically adjust the factor. Values between :math:`10^{-2}` and :math:`10^{-4}` have shown to yield satisfying results. `[1]`_

Evaluation on Qiskit's :class:`QasmSimulator <qiskit_aer.QasmSimulator>` now yields less variance in the model, as depicted in :numref:`fig_qnn_output_low_var`.

.. _fig_qnn_output_low_var:
.. figure:: ../_static/qnn/qnn_output_low_var.svg
    :alt: QNN Output with low variance
    :width: 600
    :align: center

    Logarithm and output of :class:`QNNRegressor` :math:`f(\theta, x)`, trained with variance regularization, evaluated on Qiskit's :class:`QasmSimulator <qiskit_aer.QasmSimulator>`. The QNN output has a low variance.

.. rubric:: References

_`[1]` D. A. Kreplin and M. Roth "Reduction of finite sampling noise in quantum neural networks". `arXiv:2306.01639 <https://arxiv.org/abs/2306.01639>`_ (2023).
