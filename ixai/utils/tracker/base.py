# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>
#          Rohit Jagtani

import abc
import numpy as np


class Tracker(abc.ABC):
    """Base Tracker

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abc.abstractmethod
    def __init__(self):
        self.tracked_value = 0
        self.N = 0

    def __repr__(self):
        return f"{np.round(self.tracked_value, 2)}"

    def __call__(self, *args, **kwargs):
        """Returns the current tracked value"""
        return self.tracked_value

    def get(self):
        """Returns the current tracked value"""
        return self.tracked_value

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> "Tracker":
        """Updates the tracker with a new value or values."""
        raise NotImplementedError

    def get_normalized(self):
        """Default returns the current tracked value"""
        return self.get()
