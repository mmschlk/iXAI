"""Test for trackers."""
import pytest
import numpy as np

from ixai.utils.tracker import WelfordTracker, ExponentialSmoothingTracker

dummy_value_stream_1 = [i for i in range(1, 1001)]
dummy_value_stream_2 = [0 for i in range(1, 1001)]


@pytest.mark.parametrize("parameters", [(0.01, 901.0042739534927), (1., 1000.), (0., 0.)])
def test_tracker_exponential_smoothing(parameters):
    alpha, result_1 = parameters
    tracker = ExponentialSmoothingTracker(alpha=alpha)
    for value in dummy_value_stream_1:
        tracker.update(value)
    assert tracker.get() == pytest.approx(result_1)


@pytest.mark.parametrize("data_stream", [dummy_value_stream_1, dummy_value_stream_2])
def test_tracker_welford(data_stream):
    values = data_stream
    mean = np.mean(values)
    std = np.std(values)
    var = np.var(values)
    tracker = WelfordTracker()
    for value in data_stream:
        tracker.update(value)
    assert tracker.get() == pytest.approx(mean)
    assert tracker.mean == pytest.approx(mean)
    assert tracker.var == pytest.approx(var)
    assert tracker.std == pytest.approx(std)
    assert tracker.N == len(data_stream)
    