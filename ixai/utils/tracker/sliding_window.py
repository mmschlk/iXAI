from typing import Union

import numpy as np

from ixai.utils.tracker.base import Tracker


class SlidingWindowTracker(Tracker):
    """A sliding window tracker that stores the k last elements of the stream allowing for computation of the mean and
    variance of the last k values."""

    def __init__(self, k: int):
        assert 0 < k, "The 'window_size' must be greater than zero."
        self.window_k = 0
        self.k = k
        self.sliding_window = np.array([np.NaN for _ in range(self.k)])

    def update(self, value_i: Union[int, float]) -> "Tracker":
        """Adds one value to the Tracker

        Args:
            value_i (int or float): The numeric value to be added to the tracker.
        """
        if self.window_k < self.k:
            self.sliding_window[self.window_k] = value_i
            self.window_k += 1
        else:
            self.window_k = 0
            self.sliding_window[self.window_k] = value_i
        return self

    def __call__(self, *args, **kwargs):
        """Returns the current mean of the sliding window."""
        return self.mean

    def __repr__(self):
        return f"{round(self.mean, 2)}"

    @property
    def mean(self):
        """Returns the current mean of the sliding window."""
        return float(np.nanmean(self.sliding_window, axis=0))

    @property
    def var(self):
        """Returns the variance of the sliding window."""
        return float(np.nanvar(self.sliding_window, axis=0))

    @property
    def std(self):
        """Returns the standard deviation of the sliding window."""
        return float(np.nanstd(self.sliding_window, axis=0))
