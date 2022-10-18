from .base_storage import BaseStorage


class ReservoirStorage(BaseStorage):
    """Reservoir Storage - base class
    """

    def __init__(
            self,
            store_targets: bool,
            size: int
    ):
        self.size = size
        self.store_targets = store_targets
