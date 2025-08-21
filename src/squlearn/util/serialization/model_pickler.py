import numpy as np
import dill
import sympy as sp

from qiskit.quantum_info import SparsePauliOp

from ..executor import Executor

EXECUTOR_ID = "squlearn.util.executor"


class ExecutorPickler(dill.Pickler):
    """
    Custom Pickler to handle the `Executor` object.
    When an `Executor` is encountered, it will not be serialized directly.
    Instead, it will return a marker that can be used to reconstruct the `Executor` later.
    """

    def __init__(self, file):
        super().__init__(file)

    def persistent_id(self, obj):
        """
        This method is called when an object is being serialized.
        If the object is an instance of `Executor`, it returns a marker that can be used to reconstruct the `Executor` later.
        Otherwise, it returns None, allowing the default serialization behavior to take place.
        """

        if isinstance(obj, Executor):
            return (EXECUTOR_ID, None)

        if isinstance(obj, SparsePauliOp):
            try:
                # Qiskit stores coefficients in a private array `_coeffs`
                coeffs = getattr(obj, "_coeffs", None)
                if coeffs is not None:

                    if isinstance(coeffs, np.ndarray) and coeffs.dtype == object:

                        # Convert each coefficient to a SymPy object that is picklable in contrast
                        # To the default qiskit object
                        obj._coeffs = np.array(
                            [
                                sp.sympify(c._symbol_expr) if hasattr(c, "_symbol_expr") else c
                                for c in coeffs
                            ],
                            dtype=object,
                        )
            except Exception:
                # If anything goes wrong, fall back to default serialization.
                pass

        return None


class ExecutorUnpickler(dill.Unpickler):
    """
    Custom Unpickler to handle the `Executor` object.
    When a marker for an `Executor` is encountered, it will return the `Executor` instance that was provided during initialization.
    """

    def __init__(self, file, executor: Executor):
        super().__init__(file)
        self._executor = executor

    def persistent_load(self, pid):
        """
        This method is called when an object is being deserialized.
        If the persistent ID indicates that it is an `Executor`, it returns the `Executor` instance.
        Otherwise, it calls the superclass method to handle the default deserialization behavior.
        """

        type_tag, _ = pid
        if type_tag == EXECUTOR_ID:
            return self._executor
        return super().persistent_load(pid)
