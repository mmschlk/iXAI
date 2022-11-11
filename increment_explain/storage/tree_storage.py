import random
import typing

import numpy as np
from river import base

from .interval_storage import IntervalStorage
from .geometric_reservoir_storage import GeometricReservoirStorage

from .base import BaseStorage
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from river.tree import HoeffdingAdaptiveTreeClassifier, HoeffdingAdaptiveTreeRegressor, HoeffdingTreeRegressor, HoeffdingTreeClassifier
from river.ensemble import AdaptiveRandomForestRegressor, AdaptiveRandomForestClassifier

from ..utils.trackers import WelfordTracker
from river.metrics import R2, Accuracy
from river.utils import Rolling


def walk_through_tree(node, x_i, until_leaf=True) -> typing.Iterable[typing.Union["Branch", "Leaf"]]:
    """Iterate over the nodes of the path induced by x_i."""
    yield node
    try:
        yield from walk_through_tree(node.next(x_i), x_i, until_leaf)
    except KeyError:
        if until_leaf:
            children = node.children
            ratios = [child.total_weight for child in children]
            node = random.choices(children, weights=ratios, k=1)[0]
            yield node
            yield from walk_through_tree(node, x_i, until_leaf)
    except AttributeError:  # we are at a leaf node -> which was already returned
        pass


class MeanVarRegressor(base.Regressor):

    def __init__(self):
        self.stat_object = WelfordTracker()

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        mean = self.stat_object.mean
        std = self.stat_object.std
        prediction = np.random.normal(loc=mean, scale=abs(std), size=1)[0]
        return prediction

    def learn_one(self, x: dict, y: base.typing.RegTarget) -> "base.Regressor":
        y_i = float(y)
        self.stat_object.update(value_i=y_i)
        return self


class TreeStorage(BaseStorage):
    """ A Tree Storage that trains a decsion tree for each feature.
    """

    def __init__(self, cat_feature_names: List[str], num_feature_names: List[str]):
        self.feature_names = cat_feature_names + num_feature_names
        self.cat_feature_names = cat_feature_names
        self.num_feature_names = num_feature_names
        self._storage_x = {cat_feature: HoeffdingAdaptiveTreeClassifier(max_depth=5, leaf_prediction='nba', binary_split=True)
                           for cat_feature in self.cat_feature_names}
        self._storage_x.update(
            {num_feature: HoeffdingAdaptiveTreeRegressor(max_depth=5, leaf_prediction='adaptive', binary_split=True)
             for num_feature in self.num_feature_names}
        )
        self.performances = {num_feature: Rolling(R2(), window_size=1000) for num_feature in self.num_feature_names}
        self.performances.update(
            {cat_feature: Rolling(Accuracy(), window_size=1000) for cat_feature in self.cat_feature_names}
        )
        self.data_reservoirs = {feature: {} for feature in self.feature_names}

    def update(self, x: Dict, y: Optional[Any] = None):
        """Given a data point, it updates the storage.
        Args:
            x: Features as List of Dicts
            y: Target as float or integer
        Returns:
            None
        """
        for feature_name, feature_value in x.items():
            if feature_name in self.feature_names:
                feature_model = self._storage_x[feature_name]
                x_i = {**x}
                y_i = x_i.pop(feature_name)
                feature_model.learn_one(x_i, y_i)
                self._update_data_reservoirs(feature_name, feature_model, x_i, x)
                pred_i = feature_model.predict_one(x_i)
                self.performances[feature_name].update(y_i, pred_i)

    @staticmethod
    def get_path_through_tree(feature_model, x_i):
        walked_path = ''
        path = iter(walk_through_tree(feature_model._root, x_i, until_leaf=True))
        for stop in path:
            walked_path += str(stop) + "|"
            if hasattr(stop, 'repr_split'):
                walked_path += str(stop.repr_split) + "|"
                branch_no = stop.branch_no(x_i)
                walked_path += str(branch_no) + "|"
        return walked_path

    def _update_data_reservoirs(self, feature_name, feature_model, x_i, x):
        data_reservoir = self.data_reservoirs[feature_name]
        leaf_id = self.get_path_through_tree(feature_model, x_i)
        if leaf_id not in data_reservoir:
                data_reservoir[leaf_id] = GeometricReservoirStorage(size=10, store_targets=False, constant_probability=1.0)
            #data_reservoir[leaf_id] = IntervalStorage(size=10, store_targets=False)
        data_reservoir[leaf_id].update(x)

    def __call__(self, feature_name: str) -> Tuple[Union[HoeffdingTreeRegressor, HoeffdingTreeClassifier], str]:
        if feature_name in self.cat_feature_names:
            return self._storage_x[feature_name], 'cat'
        elif feature_name in self.num_feature_names:
            return self._storage_x[feature_name], 'num'
        else:
            raise ValueError(f"The {feature_name} is not stored.")
