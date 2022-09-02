"""
This module gathers converter classes and functions to transform river models into sklearn-like functions and
vice versa.
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>

import numpy as np
from typing import Union, Sequence
from river.base import Regressor, Classifier
from river import stream


class RiverToPredictionFunction:

    def __init__(
            self,
            river_model: Union[Regressor, Classifier],
            feature_names: list[str]
    ) -> None:
        self.river_model = river_model
        self.feature_names = feature_names

    def predict(self, x: Sequence) -> np.ndarray:
        y_pred = np.empty(shape=len(x))
        if isinstance(x, np.ndarray):
            for i, (x_i, _) in enumerate(stream.iter_array(x, feature_names=self.feature_names)):
                if i > len(x):
                    break
                y_pred[i] = self.river_model.predict_one(x_i)
        else:
            # TODO implement for non-numpy x inputs also a fast prediction
            raise NotImplementedError("Currently only numpy arrays can be used for prediction.")
        return y_pred
