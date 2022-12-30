import pytest
import numpy as np
from river.tree import HoeffdingAdaptiveTreeClassifier, HoeffdingAdaptiveTreeRegressor

from ixai.storage import (BatchStorage, IntervalStorage,
                          SequenceStorage,
                          UniformReservoirStorage,
                          GeometricReservoirStorage, TreeStorage)


@pytest.fixture
def dummy_stream_small():
    stream = []
    for i in range(10):
        stream.append(({'t': i}, i))
    return stream


@pytest.fixture
def dummy_cat_stream():
    np.random.seed(42)
    feature_names = ['cat_1', 'cat_2', 'num_1']
    cat_feature_names = ['cat_1', 'cat_2']
    num_feature_names = ['num_1']
    x = np.zeros(shape=(1000, 3))
    x[:, 2] = 100.
    x[:250, 0] = 1
    x[250:500, 0] = 2
    x[500:750, 0] = 3
    x[750:, 0] = 4
    x[x[:, 0] >= 3, 1] = 1
    np.random.shuffle(x)
    stream = []
    for t in range(len(x)):
        stream.append({feature_names[i]: x[t][i] for i in range(len(feature_names))})
    return stream, feature_names, cat_feature_names, num_feature_names


@pytest.mark.parametrize("store_targets", [True, False])
def test_batch_storage(dummy_stream_small, store_targets):
    storage = BatchStorage(store_targets=store_targets)
    for x_i, y_i in dummy_stream_small:
        storage.update(x_i, y_i)
    assert len(storage) == 10
    if store_targets:
        assert len(storage.get_data()[1]) == 10
    else:
        assert len(storage.get_data()[1]) == 0


@pytest.mark.parametrize("store_targets", [True, False])
@pytest.mark.parametrize("size", [5, 8, 1])
def test_interval_storage(dummy_stream_small, store_targets, size):
    storage = IntervalStorage(store_targets=store_targets, size=size)
    for x_i, y_i in dummy_stream_small:
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
def test_sequence_storage(dummy_stream_small, store_targets):
    storage = SequenceStorage(store_targets=store_targets)
    for x_i, y_i in dummy_stream_small:
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
def test_uniform_reservoir_storage(dummy_stream_small, store_targets, size):
    storage = UniformReservoirStorage(store_targets=store_targets,
                                      size=size)
    for x_i, y_i in dummy_stream_small:
        storage.update(x_i, y_i)
    assert len(storage) == size
    if store_targets:
        assert len(storage.get_data()[1]) == size
    else:
        assert len(storage.get_data()[1]) == 0


@pytest.mark.parametrize("store_targets", [True, False])
@pytest.mark.parametrize("size", [3, 5])
def test_geometric_reservoir_storage(dummy_stream_small, store_targets, size):
    storage = GeometricReservoirStorage(store_targets=store_targets,
                                        size=size)
    for x_i, y_i in dummy_stream_small:
        storage.update(x_i, y_i)
    assert len(storage) == size
    if store_targets:
        assert len(storage.get_data()[1]) == size
    else:
        assert len(storage.get_data()[1]) == 0


def test_tree_storage(dummy_cat_stream):
    stream, feature_names, cat_feature_names, num_feature_names = dummy_cat_stream
    storage = TreeStorage(cat_feature_names=cat_feature_names,
                          num_feature_names=num_feature_names,
                          max_depth=3,
                          leaf_reservoir_length=100,
                          grace_period=10,
                          seed=42)
    assert len(storage) == 0

    for x_i in stream:
        storage.update(x=x_i)

    assert len(storage.feature_names) == 3
    assert len(storage.cat_feature_names) == 2
    assert len(storage.num_feature_names) == 1
    assert len(storage.data_reservoirs) == 3
    assert len(storage.performances) == 3
    assert storage._leaf_reservoir_length == 100
    assert len(storage._storage_x) == 3
    assert len(storage) == 1000

    for feature_name in feature_names:
        feature_model, feature_type = storage(feature_name)
        if feature_type == 'cat':
            assert type(feature_model) == HoeffdingAdaptiveTreeClassifier
        else:
            assert type(feature_model) == HoeffdingAdaptiveTreeRegressor
        data_reservoirs = storage.data_reservoirs[feature_name]
        assert len(data_reservoirs) > 0

    try:
        storage('cat_99')
    except ValueError:
        assert True
