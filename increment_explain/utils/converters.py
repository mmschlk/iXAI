"""
This module gathers converter classes and functions to transform river models into sklearn-like functions and
vice versa.
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>

import numpy as np
from typing import Union, Sequence, Callable
from river.base import Regressor, Classifier
from river import stream


class RiverToPredictionFunction:

    def __init__(self,
                 river_model: Union[Regressor, Classifier],
                 feature_names: list[str],
                 classification: bool = True
                 ) -> None:
        self.river_model = river_model
        self.feature_names = feature_names
        self.classification = classification

    def predict(self, x: Sequence) -> np.ndarray:
        if isinstance(x, np.ndarray):
            y_pred = np.empty(shape=len(x))
            for i, (x_i, _) in enumerate(stream.iter_array(x, feature_names=self.feature_names)):
                if i > len(x):
                    break
                if self.classification:
                    y_pred[i] = self.river_model.predict_proba_one(x_i)[1]
                else:
                    y_pred[i] = self.river_model.predict_one(x_i)[1]
        elif isinstance(x, dict):
            y_pred = np.empty(shape=1)
            if self.classification:
                y_pred[0] = self.river_model.predict_proba_one(x)[1]
            else:
                y_pred[0] = self.river_model.predict_one(x)
        elif isinstance(x, list):
            y_pred = np.empty(shape=len(x))
            for i, x_i in enumerate(x):
                if self.classification:
                    y_pred[i] = self.river_model.predict_proba_one(x_i)[1]
                else:
                    y_pred[i] = self.river_model.predict_one(x_i)[1]
        else:
            # TODO implement for non-numpy x inputs also a fast prediction
            raise NotImplementedError("Currently only numpy arrays can be used for prediction.")
        return y_pred


class PredictionFunctionToRiverInput:

    def __init__(self,
                 prediction_function: Callable,
                 ) -> None:
        self.prediction_function = prediction_function

    def __call__(self, x: Union[dict[str], np.ndarray]):
        if isinstance(x, dict):
            x_list = list(x.values())
            y_pred = np.asarray([self.prediction_function(x_list)])
        else:
            y_pred = np.empty(shape=len(x))
            for i in range(len(x)):
                y_pred[i] = self.prediction_function(x[i])
        return y_pred
