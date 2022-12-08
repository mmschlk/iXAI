from typing import Union

from increment_explain.utils.tracker.base import Tracker


class ExponentialSmoothingTracker(Tracker):
    """A Tracker that applies Exponential Smoothing on the numeric input values"""

    def __init__(self, alpha: float):
        assert 0 <= alpha <= 1, "Alpha must be set to a value in between zero and one. [0,1]."
        super().__init__()
        self.alpha = alpha

    def update(self, value_i: Union[int, float]) -> "Tracker":
        """Adds one value to the Tracker

        Parameters
        ----------
        value_i : number (int or float)
            numeric value to be added to the tracker
        """
        self.tracked_value = (1 - self.alpha) * self.tracked_value + self.alpha * value_i
        self.N += 1
        return self
