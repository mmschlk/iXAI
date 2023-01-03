import typing
import warnings

from ixai.utils.wrappers.base import Wrapper
from ixai.utils.wrappers import SklearnWrapper, TorchWrapper, RiverWrapper


def validate_model_function(model_function: typing.Any) -> typing.Callable:
    if isinstance(model_function, Wrapper):
        return model_function  # we assume the wrapper is applied correctly

    function_name = str(type(model_function.__self__))
    if 'sklearn' in function_name:
        return SklearnWrapper(prediction_function=model_function)
    elif 'river' in function_name:
        return RiverWrapper(prediction_function=model_function)
    elif 'torch' in function_name:
        warnings.warn("Torch Model Function provided. Default device 'cpu' is used. If your module is not running on "
                      "this device. Please manually apply the `ixai.utils.wrappers.TorchWrapper` and specify the "
                      "correct device.", UserWarning)
        return TorchWrapper(link_function=model_function)
    else:  # no known model function
        warnings.warn("The provided model function cannot be automatically wrapped and will be used directly. "
                      "If you follow the internal logic of our model functions and designed your model function "
                      "accordingly, this warning can be neglected. If not please investigate how the model functions "
                      "need to be designed.", UserWarning)
    return model_function
