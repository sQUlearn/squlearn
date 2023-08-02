.. _quantum_kernel_methods:

.. currentmodule:: squlearn.kernel

######################
Quantum Kernel Methods
######################

Quantum Kernel Methods are amongst the most promising approaches to (supervised) Quantum Machine 
Learning (QML). This is due to the fact that it has been shown [1] that they can be formally 
embedded into the rich mathematical framework of classical kernel theory. The key idea in kernel 
theory is to solve the general task of machine learning, i.e. finding and studying patterns in data, 
in a high dimensional feature space - the reproducing kernel Hilbert space (RKHS) - where the 
original learning problem attains a trivial form. The mapping from the original space to the RKHS 
(which in general can be infinite-dimensional) is done by so called feature maps. The RKHS is 
endowed with an inner product which provides access to the high dimensional space without the need 
to ever explicitly calculate the high- (infinite-) dimensional feature vecotrs. This is known as 
the *kernel trick*: the feature map and the inner product define a kernel function and vice versa, 
via

:math:`K(x,y) = \Braket{\phi(x), \phi(y)}`

Due to the inner product, the kernel function formally computes the distance betweed data points
:math:`x` and :math:`y` and thus effectively reduces to the illustrative interpretation of a 
similarity measure between data points.

The key point of Quantum Kernel methods is that they can be fundamentally formulated as a classical 
kernel method whose kernel is computed by a quantum computer. By Using a quantum computer for the 
calculation of the kernel one naturally exploits the inherent quantum mechanical phenomena of 
*superposition* and *entanglement* - a fact which holds out the prospect of designing machine 
learning models that are able deal with complex problems that are out of reach for conventional 
machine learning methods.

Quantum Kernel methods work analougsly to their classical counterparts, with data embedded into 
the exponentially increasing quantum Hilbert space via a quantum feature map

:math:`\ket{\psi(x,\boldsymbol{\theta})} = U_{\boldsymbol{\theta}}(x)\ket{0}`

with :math:`U_{\boldsymbol{\theta}}(x)` a parametrized quantum circuit for data enconding applied 
to the initial qubit state :math:`\ket{0}`. At this, we can consider the space of complex matrices

:math:`\rho(x,\boldsymbol{\theta})=\ket{\psi(x,\boldsymbol{\theta})}\bra{\psi(x,\boldsymbol{\theta})}`

enriched with the *Hilbert-Schmidt inner product* as the feature space of a quantum model and 
state [1]. Thus, in quantum computing, access to the Hilbert space of quantum states is given by 
measurements. In contrast to conventional kernel methods, the central task for quantum kernel 
methods is to find sufficiently well suited quantum feature maps for a given data set, while in 
classical kernel methods one never only treats feature maps implicitly by selecting a proper
kernel function for a given problem.


.. currentmodule:: squlearn.kernel.ml

High-Level methods that employ quantum kernels
----------------------------------------------
In general, kernel methods refer to a collection of pattern analysis algorithms that use kernel 
functions to operate in a high-dimensional feature space. Probably, the most famous representative
of these kernel-based algorithms is *Support Vector Machines (SVMs)*. Kernel methods are most 
commonly used in a supervised learning framework for either classification or regression tasks. 
But, there are use-cases for kernel-based unsupervised algorithms too; however, these are (currently)
not covered by sQUlearn. 

In the NISQ era (no access to fast linear algebra algorithms such as HHL), the basic notion of 
quantum kernel methods is to merely compute the kernel matrix with a quantum computer and 
subsequently pass it to an conventional kernel algorithms. For this task, sQUlearn provides a 
convenient workflow by either wrapping the corresponding scikit-learn estimators or by 
independetly implementing them analougsly, adapted to the needs of quantum kernel methods. 
sQUlearn offers SVMs for both classifciation (QSVC) and regression (QSVR) tasks, Gaussian 
Processes (GPs) for both classification (QGPC) and regression (QGPR) tasks as well as a 
quantum kernel ridge regression routine (QKRR).

Classification
##############

In terms of classification tasks, sQUlearn provides:

.. autosummary::
   :nosignatures:

   QSVC
   QGPC

We refer to the documentations and examples of the respective methods for in-depth information
and user guidelines.

Regression
##########

In terms of regression tasks, sQUlearn provides:

.. autosummary::
   :nosignatures:

   QSVR
   QKRR
   QGPR

We refer to the documentations and examples of the respective methods for in-depth information
and user guidelines.


Methods to evaluate quantum kernels
-----------------------------------
sQUlearn provides two methods to evaluate quantum kernels, which also represent the 
standard approaches to quantum kernel methods in the literature.

Central to both approaches is the embedding of data into the quantum Hilbert space by using
quantum feature maps, which are nothing but parameterized quantum circuits. At this, the 
parametrization is optional, but can be used for optimally adjusting a resulting quantum kernel 
to a given data set.

.. currentmodule:: squlearn.kernel.matrix

Fidelity Quantum Kernel (FQK) via :class:`FidelityKernel`
#########################################################

The straightforward and natural way to the definition of a quantum kernel [1-4], is to use the 
*native geometry of the quantum state space* which is inherent in the *Hilbert-Schmidt inner product*, 
i.e.

:math:`k^Q(x,x^\prime)=\mathrm{tr}[\rho(x,\boldsymbol{\theta})\rho(x^\prime,\boldsymbol{\theta})]` 

which, for pure states, reduces to

:math:`k^Q(x,x^\prime)=\left|\Braket{\psi(x,\boldsymbol{\theta})|\psi(x^\prime, \boldsymbol{\theta})}\right|^2`,

i.e. a *fidelity-type metric*. From this, it immediately follows that evaluating the FQK scales 
quadratically in the dataset size. Therefore, the applicability of FQKs is naturally restricted to small dataset
instances.

In sQUlearn a FQK (instance) can be defined as shown by the following example:

.. code-block:: python 

    from squlearn.util import Executor
    from squlearn.feature_map import ChebPQC
    from squlearn.kernel import FidelityKernel
    fmap = ChebPQC(num_qubits=4, num_features=1, num_layers=2)
    fqk_instance = FidelityKernel(
        feature_map=fmap,
        executor=Executor('statevector_simulator')
    )

**Add regularization example, as well as comment on mit_depol_noise**

Projected Quantum Kernel (PQK) via :class:`ProjectedQuantumKernel`
##################################################################

Several works show that with increasing problem size, the FQK approach suffers from 
*exponential concentration* leading to quantum models that suffer from untrainability. 
To circumvent this problem, Ref. [4] introduced a family of *projected quantum kernels*
as a solution. These work by projecting the quantum states to an approximate classical
representation by using, e.g., reduced physical observables. 

The default PQK implementation of sQUlearn is one of the simplest forms of PQKs. They work by 
measuring the one-particle reduced density matrix (1-RDM) on all qubits for the encoded state
and define the kernel as (RBF-inspired)

:math:`k^{PQ}(x,x^\prime)=\exp\left(-\gamma\sum_k\sum_{P\in\lbrace X,Y,Z\rbrace}\left[\mathrm{tr}(P\rho(x,\boldsymbol{\theta})_k) - \mathrm{tr}(P\rho(x^\prime,\boldsymbol{\theta})_k)\right]^2\right)`

where :math:`\rho(x,\boldsymbol{\theta})_k` refers to the 1-RDM, 
which is the partial trace of the quantum state 
:math:`\rho(x,\boldsymbol{\theta})=\ket{\psi(x,\boldsymbol{\theta})}\bra{\psi(x,\boldsymbol{\theta})}`
over all qubits except for the :math:`k`-th qubit. After some lines of algebra, it can be seen that
these :math:`\mathrm{tr}` arguments are nothing but expectation values for measuring the Paulis 
:math:`X,Y,Z` on each qubit in the state :math:`\ket{\psi(x,\boldsymbol{\theta})}` and thus can be 
viewed as QNNs. The definition of PQKs is ambiguous. This concerns the outer form of the kernel, i.e. 
the function into which the QNN is put, the choice measurement operators used to evaluate the QNN 
as well as their respective locallity, which eventually reflects in the order of RDMs used in the 
definition. Currently, sQUlearn implements different outer forms :math:`f(\cdot)`, which represent 
standard scikit-learn kernel functions (`Gaussian`, `Matern`, `ExpSineSquared`, `RationalQuadratic`,
`DotProduct`, `PariwiseKernel`), i.e. generally speaking, sQUlearn provides PQKS of the form

:math:`k^{PQ}(x,x^\prime) = f\left[QNN(x), QNN(x^\prime)\right]`

A respective PQK (instance) referring to the definition given above is defined as illustrated by 
the following example:

.. code-block:: python

    from squlearn.util import Executor
    from squlearn.feature_map import ChebPQC
    from squlearn.kernel import ProjectedQuantumKernel
    fmap = ChebPQC(num_qubits=4, num_features= 1, num_layers=2)
    pqk_instance = ProjectedQuantumKernel(
        feature_map=fmap,
        executor=Executor('statevector_simulator'),
        measurement='XYZ',
        outer_kernel='gaussian'
    )

Moreover, the QNNs can be evaluated with respect to different measurement operators, where in
addition to the default setting - :code:`measurement='XYZ'` - one can specify :code:`measurement='X'`, 
:code:`measurement='Y'` and :code:`measurement='Z'` for one-qubit measurements with respect to only 
one Pauli operator. Beyond that, one can also specify an operator or a list of operators, see the 
respective examples in :class:`ProjectedQuantumKernel` or the "Operators for expectation values" 
user guide.

**Add PQK regularization example**

Training of quantum kernels
---------------------------

As mentioned above, the definition of quantum kernels (both FQK and PQK) relies quantum feature maps 
that are represented through parametrized quantum circuits (PQC). This results in quantum kernels 
that contain trainable parameters to obtimally adjust to a given learning problem. The trainable
parameters are obtained from classical optimization loops which attempt to minimize a given loss 
function.

sQUlearn implements the kernel target alignment procedure as well as the Negative-Log-Likelihood. 
At the same time it provides several optimizers such as :code:`Adam` and :code:`SLSQP`; cf. 
:class:`squlearn.kernel.optimization.kernel_optimizer.KernelOptimizer` for details.

The following examples assume that you have some data set available which you previously split into 
training and test data and shows how to optimize kernels.

**Example - Kernel Target Alignment** 

    .. code-block:: python

        import numpy as np
        from qiskit.primitives import Estimator()
        from squlearn.util import Executor
        from squlearn.feature_map import ChebPQC
        from squlearn.optimizers import Adam 
        from squlearn.kernel import ProjectedQuantumKernel
        from squlearn.kernel.optimization import KernelOptimizer, TargetAlignment, NLL
        fmap = ChebPQC(num_qubits=4, num_features=1, num_layers=2)
        pqk_instance = ProjectedQuantumKernel(
            feature_map=fmap,
            executor=Executor(Estimator()),
            measurement='XYZ',
            outer_kernel='gaussian',
            parameter_seed=0
        )
        # set up the optimizer
        adam_opt = Adam(options={"maxiter":100, "lr": 0.1})
        # define KTA loss function
        kta_loss = TargetAlignment(quantum_kernel=pqk_instance)
        kta_optimizer = KernelOptimizer(loss=kta_loss, optimizer=adam_opt)
        opt_kta_result = kta_optimizer.run_optimization(x=x_train, y=y_train)
        # retrieve optimized parameters
        opt_kta_params = opt_kta_result.x
        # assign optimal kta parameters to kernel
        pqk_instance.assign_parameters(opt_kta_params)

*Note:* The same workflow has to be used for FQKs!

**Example - Negative-Log-Likelihood**

.. code-block:: python

    import numpy as np
    from qiskit.primitives import Estimator()
    from squlearn.util import Executor
    from squlearn.feature_map import ChebPQC
    from squlearn.optimizers import Adam 
    from squlearn.kernel import ProjectedQuantumKernel
    from squlearn.kernel.optimization import KernelOptimizer, TargetAlignment, NLL
    fmap = ChebPQC(num_qubits=4, num_features=1, num_layers=2)
    pqk_instance = ProjectedQuantumKernel(
        feature_map=fmap,
        executor=Executor(Estimator()),
        measurement='XYZ',
        outer_kernel='gaussian',
        parameter_seed=0
    )
    # set up the optimizer
    adam_opt = Adam(options={"maxiter":100, "lr": 0.1})
    # define NLL loss function (note that noise_val needs to bet set)
    nll_loss = NLL(quantum_kernel=pqk_instance, sigma=noise_val)
    nll_optimizer = KernelOptimizer(loss=nll_loss, optimizer=adam_opt)
    opt_nll_result = nll_optimizer.run_optimization(x=x_train, y=y_train)
    # retrieve optimized parameters
    opt_nll_params = opt_nll_params.x
    # assign optimal nll parameters to kernel
    pqk_instance.assign_parameters(opt_nll_params)

*Note:* The same workflow has to be used for FQKs!

*References*
------------

[1] `M. Schuld, "Supervised quantum machine learning models are kernel methods". arXiv:2101.11020v2 (2021). <http://arxiv.org/pdf/2101.11020v2>`_

[2] `M. Schuld and N. Killoran, "Quantum Machine Learning in feature Hilbert spaces". Phys. Rev. Lett. 112(4), 040504 (2019). <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.040504>`_

[3] `S. Jerbi et al., "Quantum machine learning beyond kernel methods". arXiv:2110.13162v3 (2023) <https://arxiv.org/abs/2110.13162>`_ 

[4] `H. Y. Huang et al., "Power of data in quantum machine learning". Nat. Commun. 12, 2631 (2021). <https://www.nature.com/articles/s41467-021-22539-9>`_ 