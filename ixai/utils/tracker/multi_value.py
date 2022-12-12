import copy

import numpy as np

from ixai.utils.tracker.base import Tracker


class MultiValueTracker(Tracker):

    def __init__(self, base_tracker: Tracker):
        super().__init__()
        self.tracked_value: list[Tracker] = [0]
        self._tracked_indices = 0
        self._base_tracker = base_tracker

    def update(self, values: list[float]) -> "Tracker":
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
            return 0.
