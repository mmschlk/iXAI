import typing
import abc
import river
from river.metrics.base import Metric


class RiverMetricToLossFunction:

    def __init__(self, river_metric: river.metrics.base.Metric):
        self._river_metric = river_metric
        self._sign = 1.
        if hasattr(self._river_metric, "bigger_is_better") and self._river_metric.bigger_is_better:
            self._sign = -1.

    @abc.abstractmethod
    def __call__(self, y_true, y_prediction):
        pass


class RiverDictLabelMetricToLossFunction(RiverMetricToLossFunction):

    def __init__(self, river_metric: river.metrics.base.Metric):
        super().__init__(river_metric)

    def __call__(self, y_true, y_prediction):
        y_prediction = {i: y_prediction[i] for i in range(0, len(y_prediction))}
        loss_i = self._river_metric.update(y_true=y_true, y_pred=y_prediction).get()
        self._river_metric.revert(y_true=y_true, y_pred=y_prediction)
        return loss_i * self._sign


class RiverSingletonMetricToLossFunction(RiverMetricToLossFunction):

    def __init__(self, river_metric: river.metrics.base.Metric):
        super().__init__(river_metric)

    def __call__(self, y_true, y_prediction):
        try:
            y_prediction = float(y_prediction[0])
        except TypeError:
            y_prediction = float(y_prediction)
        loss_i = self._river_metric.update(y_true=y_true, y_pred=y_prediction).get()
        self._river_metric.revert(y_true=y_true, y_pred=y_prediction)
        return loss_i * self._sign


def _get_loss_function_from_river_metric(river_metric: Metric):
    try:
        _ = river_metric.update(y_true=0, y_pred={0: 1}).revert(y_true=0, y_pred={0: 1})
        validated_loss_function = RiverDictLabelMetricToLossFunction(river_metric=river_metric)
        _ = validated_loss_function(0, [0])
        _ = validated_loss_function(0, [0, 0])
        return validated_loss_function
    except:
        pass
    try:
        _ = river_metric.update(y_true=0, y_pred=0).revert(y_true=0, y_pred=0)
        validated_loss_function = RiverSingletonMetricToLossFunction(river_metric=river_metric)
        _ = validated_loss_function(0, 0)
        _ = validated_loss_function(0, [0])
        return validated_loss_function
    except:
        pass
    raise ValueError(f"Provided river metric cannot be transformed into a viable loss function.")


def validate_loss_function(loss_function: typing.Union[typing.Callable, Metric], model_function: typing.Callable):
    validated_loss_function = loss_function
    if isinstance(loss_function, river.metrics.base.Metric):  # loss function is a river metric object
        validated_loss_function = _get_loss_function_from_river_metric(river_metric=loss_function)
    return validated_loss_function
