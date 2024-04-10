from typing import Union

from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ..util import Executor

from .lowlevel_qnn_base import LowLevelQNNBase
from .lowlevel_qnn_pennylane import LowLevelQNNPennyLane
from .lowlevel_qnn_qiskit import LowLevelQNNQiskit


class LowLevelQNN(LowLevelQNNBase):

    def __new__(
        self,
        pqc: EncodingCircuitBase,
        observable: Union[ObservableBase, list],
        executor: Executor,
        *args,
        **kwargs,
    ):

        if executor.quantum_framework == "pennylane":
            return LowLevelQNNPennyLane(pqc, observable, executor, *args, **kwargs)
        elif executor.quantum_framework == "qiskit":
            return LowLevelQNNQiskit(pqc, observable, executor, *args, **kwargs)
        else:
            raise RuntimeError("Quantum framework not supported")
