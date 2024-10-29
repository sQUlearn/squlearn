.. _quantum_reservoir_computing:

.. currentmodule:: squlearn.qrc

===========================
Quantum Reservoir Computing
===========================

Reservoir Computing (RC) unlike conventional deep neural network architectures, which demand weight
optimization through all layers, utilizes a specific, fixed, connected network layer (usually randomized)
called the "reservoir". This reservoir allows for a dynamic, nonlinear transformation of the input data 
into a high-dimensional feature space and then performing linear regression on it. Quantum Reservoir Computing (QRC) 
tries to take advantage of the exponential scaling dimension of the Hilbert space in the quantum computing context 
to prove an advantage over the classical RC. In QRC there is a register of accessable qubits :math:`\rho_0`, which are subject to
encoding the classical information and measurements, and hidden qubits :math:`\rho_{\mathrm{hidden}}`, which are related to the reservoir and give
the possibility for rich internal dynamics. 

First the classical input data :math:`x=\lbrace x^{(i)}\rbrace_{i=1}^D` gets related to a unitary 
evolution :math:`U(x)`, called encoding circuit, that evolves the initial state of the accessable qubits :math:`\rho_0` to 

.. math::
    \rho(x)=U(x)\rho_0U(x)^{\dagger}.

Afterwards the reservoir dynamic :math:`U_R`  gets applied to the composite of the accessable and hidden qubits, i.e

 .. math::
    \tilde{\rho}(x)=U_R(\rho(x)\otimes \rho_{\mathrm{hidden}})U_R^{\dagger}=U_R(U(x)\rho_0U(x)^{\dagger}\otimes \rho_{\mathrm{hidden}})U_R^{\dagger}
