from typing import Union

from ixai.utils.tracker.base import Tracker


class ExponentialSmoothingTracker(Tracker):
    """A Tracker that applies Exponential Smoothing on the numeric input values."""

    def __init__(self, alpha: float, dynamic_alpha: bool = False):
        assert 0 <= alpha <= 1, "Alpha must be set to a value in between zero and one. [0,1]."
        super().__init__()
        self.alpha = alpha
        self.dynamic_alpha = dynamic_alpha
        self.dynamic_alpha_cutoff = int(1 / self.alpha)

    def update(self, value_i: Union[int, float]) -> "Tracker":
        """Adds one value to the Tracker

        Args:
            value_i (int or float): The numeric value to be added to the tracker.
        """
        if not self.dynamic_alpha or self.N >= self.dynamic_alpha_cutoff:
            self.tracked_value = (1 - self.alpha) * self.tracked_value + self.alpha * value_i
        else:
            self.tracked_value = (self.N / (self.N + 1)) * self.tracked_value + (1 / (self.N + 1)) * value_i
        self.N += 1
        #self.tracked_value /= 1 - (1 - self.alpha) ** self.N
        return self

    def __call__(self, *args, **kwargs):
        return self.tracked_value / (1 - (1 - self.alpha) ** self.N)
