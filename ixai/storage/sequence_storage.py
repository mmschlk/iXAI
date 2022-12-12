from .interval_storage import IntervalStorage


class SequenceStorage(IntervalStorage):
    """ An Interval Storage storing the last sample.
    """

    def __init__(
            self,
            store_targets: bool = True
    ):
        super().__init__(
            store_targets=store_targets,
            size=1
        )
