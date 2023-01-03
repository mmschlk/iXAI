"""
This module contains validation logic for loss functions
"""
import typing

from river.metrics.base import Metric

from ixai.utils.wrappers.river import RiverMetricToLossFunction


def _get_loss_function_from_river_metric(river_metric: "Metric"):
    try:
        _ = river_metric.update(y_true=0, y_pred=0).revert(y_true=0, y_pred=0)
        validated_loss_function = RiverMetricToLossFunction(river_metric=river_metric, dict_input_metric=False)
        _ = validated_loss_function(0, {'output': 0})
        return validated_loss_function
    except AttributeError:  # river metric expects a dict as y_pred
        _ = river_metric.update(y_true=0, y_pred={0: 0}).revert(y_true=0, y_pred={0: 0})
        validated_loss_function = RiverMetricToLossFunction(river_metric=river_metric, dict_input_metric=True)
        _ = validated_loss_function(0, {'output_1': 0})
        return validated_loss_function
    except Exception as error:
        raise ValueError("Provided river metric cannot be transformed into a viable loss "
                         "function.") from error


def validate_loss_function(loss_function: typing.Union[typing.Callable, "Metric"]):
    validated_loss_function = loss_function
    if isinstance(loss_function, Metric):  # loss function is a river metric object
        validated_loss_function = _get_loss_function_from_river_metric(river_metric=loss_function)
    return validated_loss_function
