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
        """
        This method is called when an object is being serialized.
        If the object is an instance of :ref:`Executor`, it returns a marker that can be used to reconstruct the :ref:`Executor` later.
        Otherwise, it returns None, allowing the default serialization behavior to take place.
        """

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
        """
        This method is called when an object is being deserialized.
        If the persistent ID indicates that it is an :ref:`Executor`, it returns the :ref:`Executor` instance.
        Otherwise, it calls the superclass method to handle the default deserialization behavior.
        """

        type_tag, _ = pid
        if type_tag == "EXECUTOR":
            return self._executor
        return super().persistent_load(pid)
