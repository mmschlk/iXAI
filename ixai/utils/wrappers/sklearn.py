import typing

import numpy as np

from ixai.utils.wrappers.base import Wrapper


class SklearnWrapper(Wrapper):
    """Wrapper for sklearn prediction functions.

    This wrapper turns any prediction function ouput into an iterable (list or np.ndarray) output. And allows
    for dict inputs.

    Examples:
        Basic usage:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> model_function = SklearnWrapper(model.predict)

        For classifiers returning probas:
        >>> model_function = SklearnWrapper(model.predict_proba)

    Note:
        If the sklearn model is trained with access to the feature names (e.g. trained on a pandas DataFrame) it will
        raise a warning, which can be suppressed.
    """

    def __init__(self, prediction_function: typing.Callable):
        self.prediction_function = prediction_function

    def __call__(self, x: typing.Union[dict, typing.Sequence]):
        if isinstance(x, dict):
            x_list = [list(x.values())]
            y_pred = np.asarray([self.prediction_function(x_list)])
        else:
            y_pred = np.empty(shape=len(x))
            for i in range(len(x)):
                x_list = [list(x[i].values())]
                y_pred[i] = self.prediction_function(x_list)
        return y_pred
