import copy
import typing

import numpy as np


class RiverToPredictionFunction:

    def __init__(self, river_predict_function: typing.Callable):
        self._river_predict_function = river_predict_function
        self.max_labels = 1

    def _reduce_dict(self, y_prediction):
        if not isinstance(y_prediction, dict):
            return np.asarray([y_prediction])
        self.max_labels = max(self.max_labels, max(y_prediction))
        y_arr = np.zeros(shape=self.max_labels + 1)
        for key, value in y_prediction.items():
            y_arr[key] = value
        return y_arr

    def __call__(self, x: list[dict]):
        x_input = copy.copy(x)
        if isinstance(x_input, dict):
            x_input = [x_input]
        predictions = [self._reduce_dict(self._river_predict_function(x_i)) for x_i in x_input]
        return predictions


class PredictionFunctionToRiverInput:

    def __init__(self, prediction_function: typing.Callable) :
        self.prediction_function = prediction_function

    def __call__(self, x: typing.Union[dict[str], np.ndarray]):
        if isinstance(x, dict):
            x_list = list(x.values())
            y_pred = np.asarray([self.prediction_function(x_list)])
        else:
            y_pred = np.empty(shape=len(x))
            for i in range(len(x)):
                y_pred[i] = self.prediction_function(x[i])
        return y_pred
