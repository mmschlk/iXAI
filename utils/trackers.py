"""
This module gathers objects to keep track of incremental / sequential data
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>

from typing import Union
import numpy as np


# TODO write tests for all Trackers
class WelfordTracker:
    """ A Tracker that applies Welford's Algorithm to estimate the mean and variance of a sequence.

    Notes
    -----
    Taken and adapted from Ian Covert's SAGE implementation.
    """
    def __init__(self):
        self.N = 0
        self.mean = 0
        self.sum_squares = 0

    def update(self, value_i: Union[int, float]):
        """Adds one value to the Tracker

        Parameters
        ----------
        value_i : number (int or float)
            numeric value to be added to the tracker
        """
        self.N += 1
        difference_1 = value_i - self.mean
        self.mean += difference_1 / self.N
        difference_2 = value_i - self.mean
        self.sum_squares += difference_1 * difference_2

    def __call__(self, *args, **kwargs):
        return self.mean

    def __repr__(self):
        return f"mean: {round(self.mean, 2)}, var: {round(self.var, 2)}"

    @property
    def var(self):
        """Returns the variance of the stream"""
        return self.sum_squares / (max(self.N, 1) ** 2)

    @property
    def std(self):
        """Returns the standard deviation of the stream"""
        return self.var ** 0.5


class ExponentialSmoothingTracker:
    """A Tracker that applies Exponential Smoothing on the numeric input values"""

    def __init__(self, alpha: float):
        assert 0 <= alpha <= 1, "Alpha must be set to a value in between zero and one. [0,1]."
        self.estimate = 0
        self.alpha = alpha

    def update(self, value_i: Union[int, float]):
        """Adds one value to the Tracker

        Parameters
        ----------
        value_i : number (int or float)
            numeric value to be added to the tracker
        """
        self.estimate = (1 - self.alpha) * self.estimate + self.alpha * value_i

    def __call__(self, *args, **kwargs):
        return self.estimate


class SlidingWindowTracker:
    """A sliding window tracker that stores the k last elements of the stream allowing for computation of the mean and
    variance of the last k values."""

    def __init__(self, k: int):
        assert 0 < k, "The 'window_size' must be greater than zero."
        self.window_k = 0
        self.k = k
        self.sliding_window = np.empty(shape=self.k)

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

    def __call__(self, *args, **kwargs):
        """Returns the current mean of the sliding window"""
        return self.mean

    @property
    def mean(self):
        """Returns the current mean of the sliding window"""
        return np.mean(self.sliding_window, axis=0)

    @property
    def var(self):
        """Returns the variance of the sliding window"""
        return np.var(self.sliding_window, axis=0)

    @property
    def std(self):
        """Returns the standard deviation of the sliding window"""
        return np.std(self.sliding_window, axis=0)
