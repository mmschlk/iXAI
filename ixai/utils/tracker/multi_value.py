import copy
import typing

from ixai.utils.tracker.base import Tracker


class MultiValueTracker(Tracker):
    """A Tracker for storing multiple values at once in the form of a dict mapping from keys to individual Trackers.

    Attributes:
        tracked_value (dict): The dictionary containing the individual trackers for each value.
    """

    def __init__(self, base_tracker: Tracker):
        """
        Args:
            base_tracker (Tracker): The tracker object to be used for each new element to be tracked.
        """
        super().__init__()
        self.tracked_value: typing.Dict[Tracker] = {}
        self._tracked_keys: typing.Set = set()
        self._base_tracker = copy.deepcopy(base_tracker)

    def update(
            self,
            values: typing.Dict[typing.Any, typing.Union[int, float]]
    ) -> "Tracker":
        """Adds one value for tracked object to the tracker.

        Note:
            Whenever the input dictionary contains a new key not stored in the Tracker, it will be added to its storage.

        Args:
            values (dict): A dictionary mapping from keys to numeric values to be added to the tracker.
        """
        keys_in_update = set(values.keys())
        for key in keys_in_update:
            try:
                self.tracked_value[key].update(values[key])
            except KeyError:
                self.tracked_value[key] = copy.deepcopy(self._base_tracker)
                self.tracked_value[key].update(values[key])
                self._tracked_keys.add(key)
        for key in self._tracked_keys - keys_in_update:
            self.tracked_value[key].update(0)  # is zero the right value to add?
        self.N += 1
        return self

    def __call__(self):
        """Returns the current tracked values."""
        return {key: self.tracked_value[key].get() for key in self._tracked_keys}

    def get_normalized(self):
        """Normalizes the tracked values by dividing them through the sum of the values."""
        tracked_values: dict = self.get()
        if len(self._tracked_keys) <= 1:
            return tracked_values
        try:
            tracked_values = {key: value / sum(tracked_values.values()) for key, value in tracked_values.items()}
        except ZeroDivisionError:
            tracked_values = {key: 0. for key in tracked_values.keys()}
        return tracked_values

    def __repr__(self):
        return f"MultiValueTracker: {self.get()}"

    def get(self) -> dict:
        """Returns the current tracked values."""
        return self()
