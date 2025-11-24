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
from squlearn.optimizers import LBFGSB


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
        qkode = QKODE(q_kernel, loss=ode_loss, optimizer=LBFGSB())

        x_train = np.linspace(0, 0.9, 9).reshape(-1, 1)
        labels = np.zeros((len(x_train), 1))
        qkode.fit(x_train, labels)

        if version.parse(scipy_version) < version.parse("1.15"):
            regressor_result = np.array(
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
            )
        elif system() == "Windows":
            regressor_result = np.array(
                [
                    0.9973879,
                    1.12113415,
                    1.25344976,
                    1.40061217,
                    1.56992748,
                    1.7561082,
                    1.96445327,
                    2.19875868,
                    2.46402581,
                ]
            )
        elif system() == "Darwin":
            regressor_result = np.array(
                [
                    0.99705777,
                    1.12080111,
                    1.25306857,
                    1.40014354,
                    1.56938517,
                    1.75553607,
                    1.96384703,
                    2.1980863,
                    2.46328302,
                ]
            )
        else:
            regressor_result = np.array(
                [
                    0.99689371,
                    1.12059708,
                    1.25287154,
                    1.39998192,
                    1.56922034,
                    1.75531624,
                    1.96360549,
                    2.19784202,
                    2.46296489,
                ]
            )

        assert qkode._loss.order_of_ode == 1
        assert np.allclose(
            qkode.predict(x_train),
            regressor_result,
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
        qkode = QKODE(q_kernel, loss=ode_loss, optimizer=LBFGSB())

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
