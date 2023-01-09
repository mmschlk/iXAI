"""Test for validators."""
import warnings

import pytest
import torch
from sklearn.tree import DecisionTreeClassifier
from river.tree import HoeffdingTreeClassifier

from ixai.utils.validators import validate_model_function
from ixai.utils.wrappers import SklearnWrapper, RiverWrapper, TorchWrapper


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Dense = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.Dense(x)
        return torch.nn.functional.softmax(x)


@pytest.fixture
def dummy_sklearn_model():
    model = DecisionTreeClassifier()
    return model.predict


@pytest.fixture
def dummy_river_model():
    model = HoeffdingTreeClassifier()
    return model.predict_one


def dummy_custom_model_function(x):
    return x


def test_model_validator_torch_module():
    nn = TestModule()
    with pytest.warns(UserWarning):
        validated_model_function = validate_model_function(nn)
        assert isinstance(validated_model_function, TorchWrapper)


def test_model_validator_sklearn(dummy_sklearn_model):
    validated_model_function = validate_model_function(dummy_sklearn_model)
    assert isinstance(validated_model_function, SklearnWrapper)


def test_model_validator_river(dummy_river_model):
    validated_model_function = validate_model_function(dummy_river_model)
    assert isinstance(validated_model_function, RiverWrapper)


def test_model_validator_custom_function():
    with pytest.warns(UserWarning):
        validated_model_function = validate_model_function(dummy_custom_model_function)
        assert not isinstance(validated_model_function, RiverWrapper)
        assert not isinstance(validated_model_function, SklearnWrapper)
        assert not isinstance(validated_model_function, TorchWrapper)
