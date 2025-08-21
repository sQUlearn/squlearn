import pytest

import numpy as np
import sympy as sp


from squlearn.util import Executor
from squlearn.encoding_circuit import ChebyshevTower
from squlearn.kernel import ProjectedQuantumKernel, FidelityKernel
from squlearn.kernel.loss import ODELoss
from squlearn.kernel import QKODE
from squlearn.optimizers import LBFGSB


class TestQKODE:

    def test_qkode_pqk(self):
        # Define the ODE
        x, f, dfdx = sp.symbols("x f dfdx")
        eq = dfdx - f

        # Initial values
        initial_values = [1]

        # Create the encoding circuit
        encoding_circuit = ChebyshevTower(num_qubits=4, num_chebyshev=2)

        # Create the ODE loss function
        ode_loss = ODELoss(eq, symbols_involved_in_ode=[x, f, dfdx], initial_values=initial_values)

        # Create the quantum kernel
        q_kernel = ProjectedQuantumKernel(
            encoding_circuit=encoding_circuit,
            executor=Executor("pennylane"),
            regularization="tikhonov",
        )

        # Create the QKODE instance
        qkode = QKODE(q_kernel, loss=ode_loss, optimizer=LBFGSB())

        x_train = np.linspace(0, 0.9, 9).reshape(-1, 1)
        labels = np.zeros((len(x_train), 1))
        qkode.fit(x_train, labels)

        assert qkode._loss.order_of_ode == 1
        assert np.allclose(
            qkode.predict(x_train),
            np.array(
                [
                    0.99663332,
                    1.12030422,
                    1.2525221,
                    1.39952847,
                    1.56866833,
                    1.75470725,
                    1.96290259,
                    2.19701171,
                    2.46206022,
                ]
            ),
            atol=1e-3,
        )

    def test_qkode_fqk(self):
        # Define the ODE
        x, f, dfdx = sp.symbols("x f dfdx")
        eq = dfdx - f

        # Initial values
        initial_values = [1]

        # Create the encoding circuit
        encoding_circuit = ChebyshevTower(num_qubits=4, num_chebyshev=2)

        # Create the ODE loss function
        ode_loss = ODELoss(eq, symbols_involved_in_ode=[x, f, dfdx], initial_values=initial_values)

        # Create the quantum kernel
        q_kernel = FidelityKernel(
            encoding_circuit=encoding_circuit,
            executor=Executor("pennylane"),
            regularization="tikhonov",
        )

        # Create the QKODE instance
        qkode = QKODE(q_kernel, loss=ode_loss, optimizer=LBFGSB())

        x_train = np.linspace(0, 0.9, 9).reshape(-1, 1)
        labels = np.zeros((len(x_train), 1))
        qkode.fit(x_train, labels)

        assert qkode._loss.order_of_ode == 1
        assert np.allclose(
            qkode.predict(x_train),
            np.array(
                [
                    0.99663332,
                    1.12030422,
                    1.2525221,
                    1.39952847,
                    1.56866833,
                    1.75470725,
                    1.96290259,
                    2.19701171,
                    2.46206022,
                ]
            ),
            atol=1e-3,
        )
