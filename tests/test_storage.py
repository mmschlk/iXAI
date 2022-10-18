import pytest
from increment_explain.storage import (BatchStorage, IntervalStorage,
                                       SequenceStorage,
                                       UniformReservoirStorage,
                                       GeometricReservoirStorage)


@pytest.fixture
def dummy_stream():
    # TODO - make stream an iterator
    stream = []
    for i in range(10):
        stream.append(({'t': i}, i))
    return stream


@pytest.mark.parametrize("store_targets", [True, False])
def test_batch_storage(dummy_stream, store_targets):
    storage = BatchStorage(store_targets=store_targets)
    for x_i, y_i in dummy_stream:
        storage.update(x_i, y_i)
    assert len(storage) == 10
    if store_targets:
        assert len(storage.get_data()[1]) == 10
    else:
        assert len(storage.get_data()[1]) == 0


@pytest.mark.parametrize("store_targets", [True, False])
@pytest.mark.parametrize("size", [5, 8, 1])
def test_interval_storage(dummy_stream, store_targets, size):
    storage = IntervalStorage(store_targets=store_targets, size=size)
    for x_i, y_i in dummy_stream:
        storage.update(x_i, y_i)
    assert len(storage) == size
    assert storage.get_data()[0][size-1]['t'] == 9
    assert storage.get_data()[0][0]['t'] == 10 - size
    if store_targets:
        assert len(storage.get_data()[1]) == size
        assert storage.get_data()[1][size-1] == 9
        assert storage.get_data()[1][0] == 10 - size
    else:
        assert len(storage.get_data()[1]) == 0


@pytest.mark.parametrize("store_targets", [True, False])
def test_sequence_storage(dummy_stream, store_targets):
    storage = SequenceStorage(store_targets=store_targets)
    for x_i, y_i in dummy_stream:
        storage.update(x_i, y_i)
    assert len(storage) == 1
    assert storage.get_data()[0][0]['t'] == 9
    if store_targets:
        assert len(storage.get_data()[1]) == 1
        assert storage.get_data()[1][0] == 9
    else:
        assert len(storage.get_data()[1]) == 0


@pytest.mark.parametrize("store_targets", [True, False])
@pytest.mark.parametrize("size", [3, 5])
def test_uniform_reservoir_storage(dummy_stream, store_targets, size):
    storage = UniformReservoirStorage(store_targets=store_targets,
                                      size=size)
    for x_i, y_i in dummy_stream:
        storage.update(x_i, y_i)
    assert len(storage) == size
    if store_targets:
        assert len(storage.get_data()[1]) == size
    else:
        assert len(storage.get_data()[1]) == 0


@pytest.mark.parametrize("store_targets", [True, False])
@pytest.mark.parametrize("size", [3, 5])
def test_geometric_reservoir_storage(dummy_stream, store_targets, size):
    storage = GeometricReservoirStorage(store_targets=store_targets,
                                        size=size)
    for x_i, y_i in dummy_stream:
        storage.update(x_i, y_i)
    assert len(storage) == size
    if store_targets:
        assert len(storage.get_data()[1]) == size
    else:
        assert len(storage.get_data()[1]) == 0
