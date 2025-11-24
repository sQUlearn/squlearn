import numpy as np
from packaging import version
from platform import system
import pytest
from scipy import __version__ as scipy_version
import sympy as sp

from squlearn.util import Executor
from squlearn.encoding_circuit import ChebyshevTower
from squlearn.kernel import ProjectedQuantumKernel, FidelityKernel
from squlearn.kernel.loss import ODELoss
from squlearn.kernel import QKODE
from squlearn.optimizers import SLSQP


@pytest.fixture(params=["expr", "callable"])
def ode_loss(request):
    x, f, dfdx = sp.symbols("x f dfdx")
    eq = dfdx - f

    if request.param == "expr":
        ode = eq
        symbols = [x, f, dfdx]
    else:  # callable
        symbols = [x, f, dfdx]

        def ode(x, f, dfdx):
            return dfdx - f

    return ODELoss(
        ode_functional=ode, initial_values=[1], symbols_involved_in_ode=symbols, ode_order=1
    )


class TestQKODE:

    def test_qkode_pqk(self, ode_loss):

        # Create the quantum kernel
        q_kernel = ProjectedQuantumKernel(
            encoding_circuit=ChebyshevTower(num_qubits=4, num_chebyshev=2),
            executor=Executor("pennylane"),
            regularization="tikhonov",
        )

        # Create the QKODE instance
        qkode = QKODE(q_kernel, loss=ode_loss, optimizer=SLSQP())

        x_train = np.linspace(0, 0.9, 9).reshape(-1, 1)
        labels = np.zeros((len(x_train), 1))
        qkode.fit(x_train, labels)

        assert qkode._loss.order_of_ode == 1
        assert np.allclose(
            qkode.predict(x_train),
            np.array(
                [
                    0.99673544,
                    1.12042196,
                    1.25265393,
                    1.39967403,
                    1.56883119,
                    1.75490046,
                    1.96312582,
                    2.19726599,
                    2.46239031,
                ]
            ),
            atol=1e-3,
        )

    def test_qkode_fqk(self, ode_loss):

        # Create the quantum kernel
        q_kernel = FidelityKernel(
            encoding_circuit=ChebyshevTower(num_qubits=4, num_chebyshev=2),
            executor=Executor("pennylane"),
            regularization="tikhonov",
            use_expectation=True,
        )

        # Create the QKODE instance
        qkode = QKODE(q_kernel, loss=ode_loss, optimizer=SLSQP())

        x_train = np.linspace(0, 0.9, 9).reshape(-1, 1)
        labels = np.zeros((len(x_train), 1))
        qkode.fit(x_train, labels)

        assert qkode._loss.order_of_ode == 1
        assert np.allclose(
            qkode.predict(x_train),
            np.array(
                [
                    0.99993338,
                    1.11905726,
                    1.25272212,
                    1.40211314,
                    1.56893983,
                    1.75539996,
                    1.96415182,
                    2.19823199,
                    2.46048537,
                ]
            ),
            atol=1e-3,
        )
