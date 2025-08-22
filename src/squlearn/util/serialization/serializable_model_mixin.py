from contextlib import contextmanager
from io import IOBase
from typing import IO, Type, TypeVar, Union
from .model_pickler import ModelPickler, ModelUnpickler
from ..executor import Executor

T = TypeVar("T", bound="SerializableModelMixin")


@contextmanager
def open_maybe(path_or_buf: Union[str, IO[bytes]], mode="rb"):
    """
    Context manager to handle opening a file or using an existing file-like object.
    Args:
        path_or_buf (Union[str, IO[bytes]]): A file path as a string or an IOBase object.
        mode (str): The mode in which to open the file, default is "rb"
    Yields:
        An opened file object or the provided IOBase object.
    Raises:
        TypeError: If the provided path_or_buf is neither a string nor an IOBase object
    """

    if isinstance(path_or_buf, str):
        f = open(path_or_buf, mode)
        try:
            yield f
        finally:
            f.close()
    elif isinstance(path_or_buf, IOBase):
        # already a proper stream
        yield path_or_buf
    else:
        raise TypeError(f"Expected file path or IOBase, got {type(path_or_buf)}")


class SerializableModelMixin:
    """A mixin class that provides serialization and deserialization capabilities for model objects.
    This class uses `ExecutorPickler` and `ExecutorUnpickler` to handle the serialization of model objects
    while ensuring that the `Executor` instance is properly injected during deserialization.
    """

    def dump(self, target: Union[str, IO[bytes]]) -> None:
        """
        Serializes the model object to a file or file-like object.
        Args:
            target (Union[str, IO[bytes]]): The target file path or file-like object where the model will be serialized.
        """

        with open_maybe(target, "wb") as f:
            ModelPickler(f).dump(self)

    @classmethod
    def load(cls: Type[T], source: Union[str, IO[bytes]], executor: Executor) -> T:
        """
        Deserializes the model object from a file or file-like object, injecting the provided `Executor`.
        Args:
            source (Union[str, IO[bytes]]): The source file path or file-like object from which the model will be deserialized.
            executor (Executor): The `Executor` instance to be injected into the deserialized model.
        Returns:
            The deserialized model object with the injected `Executor`.
        """

        with open_maybe(source, "rb") as f:
            return ModelUnpickler(f, executor).load()
