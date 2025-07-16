import io
import dill
import pytest

from squlearn.util.executor import Executor
from squlearn.util.serialization.serializable_model_mixin import SerializableModelMixin


class DummyExecutorHolder(SerializableModelMixin):
    def __init__(self, executor):
        self.executor = executor


class DummyUnpickler(dill.Unpickler):
    def persistent_load(self, pid):
        return None


@pytest.fixture
def executor_obj():
    return Executor("pennylane")


def test_executor_removed_by_pickler(executor_obj):
    dummy = DummyExecutorHolder(executor_obj)

    buffer = io.BytesIO()
    dummy.dump(buffer)

    # raw load with dill, which should ignore the persistent IDs
    buffer.seek(0)
    dummpy_pickler = DummyUnpickler(buffer)
    raw = dummpy_pickler.load()

    assert hasattr(raw, "executor")
    assert raw.executor is None


def test_executor_reinjected_by_unpickler(executor_obj):
    dummy = DummyExecutorHolder(executor_obj)

    # Dump via ExecutorPickler
    buffer = io.BytesIO()
    dummy.dump(buffer)

    # unpickle with ExecutorUnpickler => executor should be injected
    buffer.seek(0)
    loaded = dummy.load(buffer, executor_obj)

    assert hasattr(loaded, "executor")
    assert loaded.executor is executor_obj


def test_multiple_executor_references(executor_obj):
    class Holder(SerializableModelMixin):
        def __init__(self, executor):
            self.level1 = DummyExecutorHolder(executor)
            self.level2 = DummyExecutorHolder(executor)

    holder = Holder(executor_obj)

    buffer = io.BytesIO()
    holder.dump(buffer)

    # raw load all executor references should be None
    buffer.seek(0)
    dummy_pickler = DummyUnpickler(buffer)
    raw = dummy_pickler.load()
    assert raw.level1.executor is None
    assert raw.level2.executor is None

    # load with ExecutorUnpickler, which should reinject the executor
    buffer.seek(0)

    loaded = holder.load(buffer, executor_obj)
    assert loaded.level1.executor is executor_obj
    assert loaded.level2.executor is executor_obj


def test_invalid_path_or_buf():
    with pytest.raises(TypeError):
        SerializableModelMixin.load(123, Executor("pennylane"))  # Invalid type for source
    with pytest.raises(TypeError):
        SerializableModelMixin.dump(123)  # Invalid type for target
    with pytest.raises(TypeError):
        SerializableModelMixin.dump(None)  # None is not a valid path or buffer
