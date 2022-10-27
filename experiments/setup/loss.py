import numpy as np
from river.datasets.base import BINARY_CLF, REG
from sage.utils import MSELoss, CrossEntropyLoss
from river.metrics import Accuracy, MAE
from river.utils import Rolling

__all__ = [
    "get_loss_function",
    "cross_entropy_loss",
    "mse_loss"
]

CROSS_ENTROPY = CrossEntropyLoss(reduction='mean')
MSE = MSELoss(reduction='mean')


def mse_loss(y_prediction, y_true):
    y_prediction = np.asarray([y_prediction])
    y_prediction = y_prediction.reshape((len(y_prediction), 1))
    y_true = np.asarray([y_true])
    y_true = y_true.reshape((len(y_true))).astype(int)
    return MSE(pred=y_prediction, target=y_true)


def cross_entropy_loss(y_prediction, y_true):
    y_prediction = np.asarray([y_prediction])
    y_prediction = y_prediction.reshape((len(y_prediction), 1))
    y_true = np.asarray([y_true])
    y_true = y_true.reshape((len(y_true))).astype(int)
    return CROSS_ENTROPY(pred=y_prediction, target=y_true)


def get_loss_function(task):
    if task == BINARY_CLF:  # TODO make task checks with River Global variables
        loss_function = cross_entropy_loss
    elif task == REG:
        loss_function = mse_loss
    else:
        raise NotImplementedError(f"No standard loss implemented for task {task}.")
    return loss_function


def get_training_metric(task, rolling_window=1000):
    if task == BINARY_CLF:
        training_metric = Accuracy()
    elif task == REG:
        training_metric = MAE
    else:
        raise NotImplementedError(f"No standard loss implemented for task {task}.")
    if rolling_window > 0:
        rolling_training_metric = Rolling(training_metric, rolling_window)
    else:
        rolling_training_metric = training_metric
    return rolling_training_metric, training_metric
