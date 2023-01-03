from typing import Union

from ixai.utils.tracker.base import Tracker


class WelfordTracker(Tracker):
    """A Tracker that applies Welford's Algorithm to estimate the mean and variance of a sequence.

    Notes:
        Taken and adapted from Ian Covert's SAGE implementation.
    """
    def __init__(self):
        super().__init__()
        self.sum_squares = 0

    def update(self, value_i: Union[int, float]):
        """Adds one value to the Tracker.

        Args:
            value_i (int or float): The numeric value to be added to the tracker.
        """
        self.N += 1
        difference_1 = value_i - self.tracked_value
        self.tracked_value += difference_1 / self.N
        difference_2 = value_i - self.tracked_value
        self.sum_squares += difference_1 * difference_2
        return self

    @property
    def var(self):
        """Returns the variance of the stream."""
        return self.sum_squares / max(self.N, 1)

    @property
    def std(self):
        """Returns the standard deviation of the stream."""
        return self.var ** 0.5

    @property
    def mean(self):
        """Returns the mean of the stream."""
        return self.tracked_value
