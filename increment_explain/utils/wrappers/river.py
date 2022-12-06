"""
This module contains River Model Wrappers to turn the output of river models into lists or arrays.
"""

import copy
import typing

import numpy as np

from increment_explain.utils.wrappers.base import Wrapper


class RiverPredictionFunctionWrapper(Wrapper):
    """Wrapper for river prediction functions.

    This wrapper turns any prediction function ouput into an iterable (list or np.ndarray) output.

    Examples:
        Basic usage:
        >>> from river.ensemble import AdaptiveRandomForestClassifier
        >>> model = AdaptiveRandomForestClassifier()
        >>> model_function = RiverPredictionFunctionWrapper(model.predict_one)

        For classifiers returning probas:
        >>> model_function = RiverPredictionFunctionWrapper(model.predict_proba_one)
    """

    def __init__(self, river_predict_function: typing.Callable):
        self._river_predict_function = river_predict_function
        self._max_labels = 1

    def _reduce_dict(self, y_prediction):
        """Transforms a prediction output into a list if it is a dictionary."""
        if not isinstance(y_prediction, dict):
            return np.asarray([y_prediction])
        self._max_labels = max(self._max_labels, max(y_prediction))
        y_arr = np.zeros(shape=self._max_labels + 1)
        for key, value in y_prediction.items():
            y_arr[key] = value
        return y_arr

    def __call__(self, x: typing.Union[typing.List[dict], dict]):
        """Runs the model and transforms the output into a list or ndarray.

        Args:
            x (Union[list, dict]): Input instance as a dictionary of feature value pairs or a list of dictionaries.
        """
        x_input = copy.copy(x)
        if isinstance(x_input, dict):
            x_input = [x_input]
        predictions = [self._reduce_dict(self._river_predict_function(x_i)) for x_i in x_input]
        return predictions
