"Low-level QNN Factory."

from typing import Callable, Union
from warnings import warn

from ...observables.observable_base import ObservableBase
from ...encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ...util import Executor

from .lowlevel_qnn_pennylane import LowLevelQNNPennyLane
from .lowlevel_qnn_qiskit import LowLevelQNNQiskit
from .lowlevel_qnn_qulacs import LowLevelQNNQulacs


class LowLevelQNN:
    """
    Low-level QNN factory, which creates the specific low-level QNN based on the quantum framework.

    Args:
        pqc (EncodingCircuitBase): The parameterized quantum circuit.
        observable (Union[ObservableBase, list]): The observable(s) to measure.
        executor (Executor): The executor for the quantum circuit.
        post_processing (Callable): Optional post processing function operating on the result dict
            after evaluate.
        *args: Additional arguments that are passed to the specific QNN.
        **kwargs: Additional keyword arguments that are passed to the specific QNN.

    Returns
        LowLevelQNNBase: The specific low-level QNN based on the quantum framework.
    """

    def __new__(
        cls,
        parameterized_quantum_circuit: EncodingCircuitBase,
        observable: Union[ObservableBase, list],
        executor: Executor,
        num_features: int,
        post_processing: Callable = None,
        *args,
        **kwargs,
    ) -> Union[LowLevelQNNPennyLane, LowLevelQNNQiskit, LowLevelQNNQulacs]:

        if executor.quantum_framework == "pennylane":
            if "primitive" in kwargs:
                if kwargs["primitive"] is not None:
                    warn("Primitive argument is not supported for PennyLane. Ignoring...")
                kwargs.pop("primitive")
            return LowLevelQNNPennyLane(
                parameterized_quantum_circuit,
                observable,
                executor,
                num_features,
                post_processing,
                *args,
                **kwargs,
            )
        elif executor.quantum_framework == "qiskit":
            return LowLevelQNNQiskit(
                parameterized_quantum_circuit,
                observable,
                executor,
                num_features,
                post_processing,
                *args,
                **kwargs,
            )
        elif executor.quantum_framework == "qulacs":
            if "primitive" in kwargs:
                warn("Primitive argument is not supported for Qulacs. Ignoring...")
                kwargs.pop("primitive")
            return LowLevelQNNQulacs(
                parameterized_quantum_circuit,
                observable,
                executor,
                num_features,
                post_processing,
                *args,
                **kwargs,
            )
        else:
            raise RuntimeError("Quantum framework not supported")
