"""
This module gathers objects to keep track of incremental / sequential data
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>
#          Rohit Jagtani

import copy
import abc
from typing import Union
import numpy as np


__all__ = [
    "WelfordTracker",
    "ExponentialSmoothingTracker",
    "SlidingWindowTracker",
    "MultiValueTracker",
    "Tracker"
]


# =============================================================================
# Base Tracker Class
# =============================================================================


class Tracker(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.tracked_value = 0
        self.N = 0

    def __repr__(self):
        return f"{np.round(self.tracked_value, 2)}"

    def __call__(self, *args, **kwargs):
        return self.tracked_value

    def get(self):
        return self.tracked_value

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_normalized(self):
        return self.get()


# =============================================================================
# Public Tracker Class
# =============================================================================


class MultiValueTracker(Tracker):

    def __init__(self, base_tracker: Tracker):
        super().__init__()
        self.tracked_value: list[Tracker] = [0]
        self._tracked_indices = 0
        self._base_tracker = base_tracker

    def update(self, values: list[float]):
        if len(values) > self._tracked_indices:
            if self._tracked_indices == 0:
                self.tracked_value: list[Tracker] = []
            for _ in range(len(values) - self._tracked_indices):
                self.tracked_value.append(copy.deepcopy(self._base_tracker))
                self._tracked_indices += 1
        for i in range(self._tracked_indices):
            self.tracked_value[i].update(values[i])
        self.N += 1
        return self

    def __call__(self):
        return np.asarray([self.tracked_value[i].get() for i in range(self._tracked_indices)])

    def get(self):
        return self()

    def get_normalized(self):
        if self._tracked_indices <= 1:
            return self.get()
        try:
            return self.get() / sum(self.get())
        except ZeroDivisionError:
            return 0.0


class WelfordTracker(Tracker):
    """ A Tracker that applies Welford's Algorithm to estimate the mean and variance of a sequence.

    Notes
    -----
    Taken and adapted from Ian Covert's SAGE implementation.
    """
    def __init__(self):
        super().__init__()
        self.sum_squares = 0

    def update(self, value_i: Union[int, float]):
        """Adds one value to the Tracker

        Parameters
        ----------
        value_i : number (int or float)
            numeric value to be added to the tracker
        """
        self.N += 1
        difference_1 = value_i - self.tracked_value
        self.tracked_value += difference_1 / self.N
        difference_2 = value_i - self.tracked_value
        self.sum_squares += difference_1 * difference_2
        return self

    @property
    def var(self):
        """Returns the variance of the stream"""
        return self.sum_squares / (max(self.N, 1) ** 2)

    @property
    def std(self):
        """Returns the standard deviation of the stream"""
        return self.var ** 0.5

    @property
    def mean(self):
        """Returns the mean of the stream"""
        return self.tracked_value


class ExponentialSmoothingTracker(Tracker):
    """A Tracker that applies Exponential Smoothing on the numeric input values"""

    def __init__(self, alpha: float):
        assert 0 <= alpha <= 1, "Alpha must be set to a value in between zero and one. [0,1]."
        super().__init__()
        self.alpha = alpha

    def update(self, value_i: Union[int, float]):
        """Adds one value to the Tracker

        Parameters
        ----------
        value_i : number (int or float)
            numeric value to be added to the tracker
        """
        self.tracked_value = (1 - self.alpha) * self.tracked_value + self.alpha * value_i
        self.N += 1
        return self


class SlidingWindowTracker:
    """A sliding window tracker that stores the k last elements of the stream allowing for computation of the mean and
    variance of the last k values."""

    def __init__(self, k: int):
        assert 0 < k, "The 'window_size' must be greater than zero."
        self.window_k = 0
        self.k = k
        self.sliding_window = np.array([np.NaN for _ in range(self.k)])

    def update(self, value_i: Union[int, float]):
        """Adds one value to the Tracker

        Parameters
        ----------
        value_i : number (int or float)
            numeric value to be added to the tracker
        """
        if self.window_k < self.k:
            self.sliding_window[self.window_k] = value_i
            self.window_k += 1
        else:
            self.window_k = 0
            self.sliding_window[self.window_k] = value_i
        return self

    def __call__(self, *args, **kwargs):
        """Returns the current mean of the sliding window"""
        return self.mean

    def __repr__(self):
        return f"{round(self.mean, 2)}"

    @property
    def mean(self):
        """Returns the current mean of the sliding window"""
        return float(np.nanmean(self.sliding_window, axis=0))

    @property
    def var(self):
        """Returns the variance of the sliding window"""
        return float(np.nanvar(self.sliding_window, axis=0))

    @property
    def std(self):
        """Returns the standard deviation of the sliding window"""
        return float(np.nanstd(self.sliding_window, axis=0))
