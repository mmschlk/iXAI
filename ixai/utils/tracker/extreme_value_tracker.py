"""
This module contains the ExtremeValueTracker.
"""
import random
from collections import deque
from typing import Optional, Any

import numpy as np

from .base import Tracker
#from base import Tracker

class ExtremeValueTracker(Tracker):
    """ Extreme Value Tracker
    """

    def __init__(
            self,
            size: int = 100,
            window_length: int = 1000,
            higher_is_better: bool = True
    ):
        super().__init__()
        self.tracked_value = 0
        self.size: int = size
        self._sort_direction = 1 if higher_is_better else -1
        self._inverse_mapping = {0: 0}
        self._time_added_to_window = {0: 100_000}
        self._time_of_current_max: int = self.size + 1
        self._storage: list = [np.inf * -1 for _ in range(self.size)]
        self._current_max_pointer: int = 0
        self._current_smallest_value = 0
        self._current_smallest_value_pointer = 0
        self._values_in_storage = 1
        self._window_length = window_length

    def _delete_current_max_from_storage(self):
        self._current_max_pointer += 1
        self._current_max_pointer %= self.size
        self._values_in_storage -= 1
        self.tracked_value = self._storage[self._current_max_pointer]
        self._values_in_storage -= 1

    def _add_new_smallest_value(self, value):
        if self._values_in_storage < self.size:
            self._current_smallest_value_pointer += 1
            self._current_smallest_value_pointer %= self.size
            self._storage[self._current_smallest_value_pointer] = value
            self._time_added_to_window[self._current_smallest_value_pointer] = self.N
            self._values_in_storage += 1

    def _add_value_to_window(self, value):
        values_in_storage_after_addition = 0
        insertion_point = 0
        for index in range(self._current_max_pointer, self._current_smallest_value_pointer + self.size):
            index %= self.size
            values_in_storage_after_addition += 1
            if self._storage[index] <= value:
                insertion_point = index
                break
        self._current_smallest_value_pointer = insertion_point
        self._current_smallest_value = value
        self._storage[insertion_point] = value
        self._time_added_to_window[insertion_point] = self.N
        self._values_in_storage = values_in_storage_after_addition

    def update(self, value):
        self.N += 1
        value *= self._sort_direction
        if self.N - self._time_added_to_window[self._current_max_pointer] >= self._window_length:
            self._delete_current_max_from_storage()
        if value < self._storage[self._current_smallest_value_pointer]:
            self._add_new_smallest_value(value)
        else:
            self._add_value_to_window(value)
            self.tracked_value = self._storage[self._current_max_pointer]

    def get(self):
        return self()

    def __call__(self, *args, **kwargs):
        """Returns current min/max value."""
        return self._storage[self._current_max_pointer] * self._sort_direction



if __name__ == "__main__":
    window_length = 4
    tracker = ExtremeValueTracker(size=5, higher_is_better=True, window_length=window_length)
    #stream = list(np.random.randint(0, 1000, size=100_000))
    stream = [7, 3, 5, 4, 10, 5, 4, 3, 2, 1, -1, 10]
    reservoir = deque(maxlen=window_length)

    for n, obs in enumerate(stream):
        tracker.update(obs)
        #reservoir.append(obs)

        max_tracker = tracker()
        print(n, max_tracker, tracker._storage)
        #max_reservoir = max(reservoir)
        #print(max_tracker)

        #if max_reservoir != max_tracker:
        #    print("error")


