from typing import IO, Type, TypeVar, Union
from squlearn.util.serialization.model_pickler import ExecutorPickler, ExecutorUnpickler
from squlearn.util.executor import Executor

T = TypeVar("T", bound="SerializableModelMixin")


class SerializableModelMixin:
    """A mixin class that provides serialization and deserialization capabilities for model objects.
    This class uses :ref:`ExecutorPickler` and :ref:`ExecutorUnpickler` to handle the serialization of model objects
    while ensuring that the :ref:`Executor` instance is properly injected during deserialization.
    """

    def dump(self, target: Union[str, IO[bytes]]) -> None:
        """
        Serializes the model object to a file or file-like object.
        Args:
            target (Union[str, IO[bytes]]): The target file path or file-like object where the model will be serialized.
        """

        if hasattr(target, "write"):
            # file-like object
            pickler = ExecutorPickler(target)
            pickler.dump(self)
        else:
            # path
            with open(target, "wb") as f:
                pickler = ExecutorPickler(f)
                pickler.dump(self)

    @classmethod
    def load(cls: Type[T], source, executor: Executor) -> T:
        """
        Deserializes the model object from a file or file-like object, injecting the provided :ref:`Executor`.
        Args:
            source (Union[str, IO[bytes]]): The source file path or file-like object from which the model will be deserialized.
            executor (Executor): The :ref:`Executor` instance to be injected into the deserialized model.
        Returns:
            The deserialized model object with the injected :ref:`Executor`.
        """
        # path
        if isinstance(source, str):
            with open(source, "rb") as f:
                obj = ExecutorUnpickler(f, executor).load()
        # file-like object
        else:
            obj = ExecutorUnpickler(source, executor).load()
        return obj
