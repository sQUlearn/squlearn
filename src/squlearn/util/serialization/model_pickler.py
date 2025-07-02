import dill
from squlearn.util.executor import Executor


class ExecutorPickler(dill.Pickler):
    """
    Custom Pickler to handle the :ref:`Executor` object.
    When an :ref:`Executor` is encountered, it will not be serialized directly.
    Instead, it will return a marker that can be used to reconstruct the :ref:`Executor` later.
    """

    def __init__(self, file):
        super().__init__(file)

    def persistent_id(self, obj):
        if isinstance(obj, Executor):
            return ("EXECUTOR", None)
        return None


class ExecutorUnpickler(dill.Unpickler):
    """
    Custom Unpickler to handle the :ref:`Executor` object.
    When a marker for an :ref:`Executor` is encountered, it will return the :ref:`Executor` instance that was provided during initialization.
    """

    def __init__(self, file, executor: Executor):
        super().__init__(file)
        self._executor = executor

    def persistent_load(self, pid):
        type_tag, _ = pid
        if type_tag == "EXECUTOR":
            return self._executor
        return super().persistent_load(pid)
