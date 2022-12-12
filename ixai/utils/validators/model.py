import typing

from ..wrappers.base import Wrapper


def validate_model_function(model_function: typing.Any) -> typing.Callable:
    if isinstance(model_function, Wrapper):
        return model_function  # we assume the wrapper is applied correctly
    # TODO validate model_function
    validated_model_function = model_function
    return validated_model_function
