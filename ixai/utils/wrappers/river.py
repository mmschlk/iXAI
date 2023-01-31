"""
This module contains River Model Wrappers to turn the output of river models into lists or arrays.
"""
import typing

from river.metrics.base import Metric

from ixai.utils.wrappers.base import Wrapper


class RiverWrapper(Wrapper):
    """Wrapper for river prediction functions.

    This wrapper turns any prediction function ouput into an iterable (list or np.ndarray) output.

    Examples:
        Basic usage:

        >>> from river.ensemble import AdaptiveRandomForestClassifier
        >>> model = AdaptiveRandomForestClassifier()
        >>> model_function = RiverWrapper(model.predict_one)

        For classifiers returning probas:

        >>> model_function = RiverWrapper(model.predict_proba_one)
    """

    def __init__(self, prediction_function: typing.Callable):
        super().__init__(prediction_function, feature_names=None)
        self._seen_labels = set()

    def _extend_dict(self, y_prediction):
        """Transforms a prediction output into a dict of outputs."""
        if isinstance(y_prediction, dict):
            return y_prediction  # TODO dicts[str, str] as y_prediction can break this
        try:
            return {self.default_label: float(y_prediction)}
        except ValueError:  # y_prediction is str
            self._seen_labels.add(y_prediction)
            output = {label: 0. for label in self._seen_labels}
            output[y_prediction] = 1.
            return output

    def __call__(self, x: typing.Union[typing.List[dict], dict]) -> typing.Union[dict, typing.List[dict]]:
        """Runs the model and transforms the output into a ``list`` or ``np.ndarray``.

        Args:
            x (Union[list, dict]): Input instance as a dictionary of feature value pairs or a list of dictionaries.
        """
        if isinstance(x, dict):
            return self._extend_dict(self._prediction_function(x))
        return [self._extend_dict(self._prediction_function(x_i)) for x_i in x]


class RiverMetricToLossFunction:
    """Wrapper that transforms a river.metrics.base.Metric into a loss function.

    This Wrapper turns metrics that expect a single value as predictions (e.g. river.metrics.MAE, or
    river.metrics.Accuracy) or metrics that expect a dictionary as predictions (e.g. river.metrics.CrossEntropy) into
    a similar interface.
    """

    def __init__(self, river_metric: "Metric", dict_input_metric: bool = False):
        """
        Args:
            river_metric ("Metric"): The river metric to be used as a loss function.
            dict_input_metric (bool): Flag if the metric expects dictionary (`True`) or single value (`False`) inputs.
                Defaults to `False` and expects single values.
        """
        self._river_metric = river_metric
        self._sign = 1.
        if hasattr(self._river_metric, "bigger_is_better") and self._river_metric.bigger_is_better:
            self._sign = -1.
        self._dict_input_metric = dict_input_metric

    def __call__(self, y_true, y_prediction: dict):
        """Calculates the loss given for a single prediction given its true (expected) value.

        Args:
            y_true (Any): The true labels.
            y_prediction (dict): The predicted values.

        Returns:
            The loss value given the true and predicted labels.
        """
        if not self._dict_input_metric:
            y_prediction = y_prediction.get('output', 0)
        loss_i = self._river_metric.update(y_true=y_true, y_pred=y_prediction).get()
        self._river_metric.revert(y_true=y_true, y_pred=y_prediction)
        return loss_i * self._sign