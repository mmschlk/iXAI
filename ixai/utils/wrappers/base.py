import abc


class Wrapper(metaclass=abc.ABCMeta):
    """Base Wrapper

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
