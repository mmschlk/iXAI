import typing

import numpy as np

from increment_explain.utils.wrappers.base import Wrapper


class GenericWrapper(Wrapper):

    def __init__(self, prediction_function: typing.Callable):
        self.prediction_function = prediction_function

    def __call__(self, x: typing.Union[dict, typing.Sequence]):
        if isinstance(x, dict):
            x_list = list(x.values())
            y_pred = np.asarray([self.prediction_function(x_list)])
        else:
            y_pred = np.empty(shape=len(x))
            for i in range(len(x)):
                y_pred[i] = self.prediction_function(list(x[i].values()))
        return y_pred
