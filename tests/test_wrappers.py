import numpy as np
import pytest
import torch
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle
from river.tree import HoeffdingAdaptiveTreeClassifier, HoeffdingAdaptiveTreeRegressor
from river.metrics import Accuracy, MSE, MAE, CrossEntropy

from ixai.utils.wrappers.river import RiverMetricToLossFunction
from ixai.utils.wrappers import TorchWrapper, SklearnWrapper, RiverWrapper


def dummy_torch_model_argmax(x: torch.Tensor):
    x_input = torch.zeros(size=(1, 5), dtype=torch.float32, device='cpu')
    x_input[:, 3] = 1.
    return torch.argmax(x_input)


def dummy_torch_model_probas(x: torch.Tensor):
    x_input = torch.zeros(size=(1, 5), dtype=torch.float32, device='cpu')
    x_input[:, 3] = 1.
    return torch.softmax(x_input, dim=-1)


def dummy_data(classification=True):
    x = np.zeros(shape=(100, 4))
    x[0:50, 0] = 1.
    x[:, 1] = 1.
    x[50:, 2] = 1.
    y = np.zeros(shape=(100,))
    if classification:
        y[x[:, 0] == 1] = 1
    else:
        y[x[:, 0] == 1] = 100.
    x, y = shuffle(x, y, random_state=42)
    return x, y


@pytest.mark.parametrize("model_function_id", ['argmax', 'probas'])
def test_torch_wrapper(model_function_id):
    test_input = {0: 0, 1: 0.5, 2: 1, 3: 1.5}
    if model_function_id == 'argmax':
        model_function = TorchWrapper(dummy_torch_model_argmax)
        output = model_function(test_input)
        assert type(output) == dict
        assert len(output) == 1
        assert output == {'output': 3.0}
    else:
        model_function = TorchWrapper(dummy_torch_model_probas)
        output = model_function(test_input)
        assert type(output) == dict
        assert len(output) == 5
        assert output[3] == pytest.approx(0.40460968)


@pytest.mark.parametrize(
    "sklearn_model",
    [('dtc', DecisionTreeClassifier(max_depth=2, random_state=42)),
     ('dtr', DecisionTreeRegressor(max_depth=2, random_state=42))]
)
def test_sklearn_wrapper(sklearn_model):
    model_function_id, sklearn_model = sklearn_model
    test_input = {0: 1, 1: 0, 2: 3, 3: 0}
    feature_names_correct = [0, 1, 2, 3]
    feature_names_half = [0, 1]
    test_input_wrong_order = {1: 0, 3: 0, 2: 3, 0: 1}
    if model_function_id == 'dtc':
        x, y = dummy_data(classification=True)
        model: DecisionTreeClassifier = sklearn_model
        model.fit(x, y)
        model_function = SklearnWrapper(model.predict)
        output = model_function(test_input)
        assert type(output) == dict
        assert len(output) == 1
        assert output == {'output': 1.0}
        model_function = SklearnWrapper(model.predict_proba)
        output = model_function(test_input)
        assert type(output) == dict
        assert len(output) == 2
        assert output == {0: 0.0, 1: 1.0}
        model_function = SklearnWrapper(model.predict_proba)
        output = model_function(test_input_wrong_order)
        assert output == {0: 1.0, 1: 0.0}
        converted_input = model_function.convert_1d_input_to_arr(test_input_wrong_order)
        assert (converted_input == np.asarray([[0, 0, 3, 1]])).all()
        converted_input = model_function.convert_2d_input_to_arr([test_input, test_input])
        assert (converted_input == np.asarray([[1, 0, 3, 0], [1, 0, 3, 0]])).all()
        model_function = SklearnWrapper(model.predict_proba, feature_names=feature_names_correct)
        converted_input = model_function.convert_1d_input_to_arr(test_input_wrong_order)
        assert (converted_input == np.asarray([[1, 0, 3, 0]])).all()
        model_function = SklearnWrapper(model.predict_proba, feature_names=feature_names_half)
        converted_input = model_function.convert_1d_input_to_arr(test_input_wrong_order)
        assert (converted_input == np.asarray([[1, 0]])).all()
    else:  # dtr
        x, y = dummy_data(classification=False)
        model: DecisionTreeRegressor = sklearn_model
        model.fit(x, y)
        model_function = SklearnWrapper(model.predict)
        output = model_function(test_input)
        assert type(output) == dict
        assert len(output) == 1
        assert output == {'output': 100.}


@pytest.mark.parametrize(
    "river_model",
    [('adtc', HoeffdingAdaptiveTreeClassifier(max_depth=2, seed=42)),
     ('adtr', HoeffdingAdaptiveTreeRegressor(max_depth=2, seed=42))]
)
def test_river_wrapper(river_model):
    model_function_id, river_model = river_model
    test_input = {0: 1, 1: 0, 2: 3, 3: 0}
    test_input_wrong_order = {1: 0, 3: 0, 2: 3, 0: 1}

    x, y = dummy_data(classification=True)
    for n in range(len(x)):
        x_i, y_i = x[n], y[n]
        x_i = {i: x_i[i] for i in range(len(x_i))}
        river_model.learn_one(x=x_i, y=y_i)

    if model_function_id == 'adtc':
        model: HoeffdingAdaptiveTreeClassifier = river_model
        x, y = dummy_data(classification=True)
        for n in range(len(x)):
            x_i, y_i = x[n], y[n]
            x_i = {i: x_i[i] for i in range(len(x_i))}
            model.learn_one(x=x_i, y=y_i)
        model_function = RiverWrapper(model.predict_one)
        output = model_function(test_input)
        assert type(output) == dict
        assert len(output) == 1
        assert output == {'output': 1.0}
        model_function = RiverWrapper(model.predict_proba_one)
        output = model_function(test_input)
        assert type(output) == dict
        assert len(output) == 2
        assert output == {0.0: 0.4734982332155477, 1.0: 0.5265017667844523}
        model_function = RiverWrapper(model.predict_proba_one)
        output = model_function(test_input_wrong_order)
        assert output == {0.0: 0.4734982332155477, 1.0: 0.5265017667844523}
    else:  # adtr
        x, y = dummy_data(classification=False)
        model: HoeffdingAdaptiveTreeRegressor = river_model
        model_function = RiverWrapper(model.predict_one)
        output = model_function(test_input)
        assert type(output) == dict
        assert len(output) == 1
        assert list(output.keys())[0] == 'output'


@pytest.mark.parametrize("river_metric", [('accuracy', Accuracy()), ('mse', MSE()), ('mae', MAE())])
def test_single_value_river_metric(river_metric):
    river_metric_id, river_metric = river_metric
    loss_function = RiverMetricToLossFunction(river_metric=river_metric, dict_input_metric=False)
    loss = loss_function(y_true=1, y_prediction={'output': 1})
    if river_metric_id == 'accuracy':
        assert loss == -1.0
    else:
        assert loss == 0
    loss = loss_function(y_true=0, y_prediction={'output': 1})
    if river_metric_id == 'accuracy':
        assert loss == 0
    else:
        assert loss == 1
    try:
        loss_function(y_true=1, y_prediction={'class_0': 0.2, 'class_1': 0.8})
        assert False
    except Exception:
        assert True


@pytest.mark.parametrize("river_metric", [('cross_entropy', CrossEntropy())])
def test_dict_value_river_metric(river_metric):
    river_metric_id, river_metric = river_metric
    loss_function = RiverMetricToLossFunction(river_metric=river_metric, dict_input_metric=True)
    loss = loss_function(y_true='class_0', y_prediction={'class_0': 1.0, 'class_1': 0.0})
    assert loss == pytest.approx(0)
    _ = loss_function(y_true='class_1', y_prediction={'class_0': 0.2, 'class_1': 0.8})
    try:
        _ = loss_function(y_true='class_1', y_prediction=0.8)
    except AttributeError:
        assert True
