import typing

from ..wrappers.base import Wrapper


def validate_model_function(model_function: typing.Any) -> typing.Callable:
    if isinstance(model_function, Wrapper):
        return model_function  # we assume the wrapper to be applied correctly
    test_input = {'test_feature_1': 1, 'test_feature_2': 2}
    # TODO handle exceptions
    try:
        test_output = model_function(test_input)
    except Exception as e:
        print(e)
        raise e
    validated_model_function = model_function
    return validated_model_function