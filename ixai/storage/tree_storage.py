"""
This module contains the TreeStorage and the MeanVarRegressor leaf classifier.
"""
import typing
from typing import Dict, Any, Optional, List, Union, Tuple
import random

import numpy as np
from river import base
from river.tree import HoeffdingAdaptiveTreeClassifier, HoeffdingAdaptiveTreeRegressor, \
    HoeffdingTreeRegressor, HoeffdingTreeClassifier
from river.metrics import R2, Accuracy
from river.utils import Rolling

from ixai.utils.tracker.welford import WelfordTracker
from .geometric_reservoir_storage import GeometricReservoirStorage
from .base import BaseStorage


NODE_SEPERATOR: str = "|STOP|"


def get_all_tree_paths(node, walked_path: str = '', paths=None) -> List[str]:
    if paths is None:
        paths = []
    try:
        children = node.children
        for branch_no, child in enumerate(children):
            child_path = walked_path + "|".join((str(node), str(node.repr_split), str(branch_no)))
            child_path += NODE_SEPERATOR
            _ = get_all_tree_paths(child, walked_path=child_path, paths=paths)
    except AttributeError:
        walked_path += str(node) + NODE_SEPERATOR
        paths.append(walked_path)
    return paths


def walk_through_tree(
        node: typing.Union["Branch", "Leaf"],
        x_i: dict,
        until_leaf: bool = True
) -> typing.Iterable[typing.Union["Branch", "Leaf"]]:
    """Traverses a decision tree given a data point, and a starting node.

    Args:
        node: Target as float or integer
        x_i (dict): Data point as Dicts.
        until_leaf (bool): Flag weather to traverse the tree until a leaf node (``True``) or
            just the next node (``False``).

    Yields:
        The next node in the tree.
    """
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
    """A simple regressor model intended to be used as a leaf model in Decision Tree Regressors.

    The Regressor keeps track of the mean and standard deviation of the incoming numerical labels and samples
    prediction values from a normal distribution according to the current mean and standard deviation.
    """

    def __init__(self):
        self._stat_object = WelfordTracker()

    def predict_one(self, x=None) -> float:
        """Predicts a value based on the current summary statistics.

        Args:
            x (Any): input features (that are not used for prediction)

        Returns:
            float: The predicted value.
        """
        mean = self._stat_object.mean
        std = self._stat_object.std
        prediction = np.random.normal(loc=mean, scale=abs(std), size=1)[0]
        return prediction

    def learn_one(self, x: Any, y: base.typing.RegTarget) -> "base.Regressor":
        """Updates the summary statistics based on the target labels.

        Args:
            x (Any): input features (that are not used for prediction)
            y (base.typing.RegTarget): A number that is transformable into a float.

        Returns:
            base.Regressor: The Regressor itself.
        """
        y_i = float(y)
        self._stat_object.update(value_i=y_i)
        return self


class TreeStorage(BaseStorage):
    """ A Tree Storage that trains incremental decision trees for each feature.

    Attributes:
        feature_names (list[str]): List of features stored.
        cat_feature_names (list[str]): List of categorical features stored.
        num_feature_names (list[str]): List of numerical features stored.
        performances (dict[Any, Union[R2, Accuracy]]): Dictionary of performance metrics per incremental decision tree
            for each feature stored.
        data_reservoirs (dict[str, dict]): Dictionary of data reservoirs for each feature and leaf nodes.
    """

    def __init__(
            self,
            cat_feature_names: list,
            num_feature_names: list,
            max_depth: int = 5,
            leaf_reservoir_length: int = 10,
            grace_period: int = 200,
            seed: Optional[int] = None
    ):
        """ A Tree Storage that trains incremental decision trees for each feature.

        Args:
            cat_feature_names (list[str]): List of categorical features to be stored.
            num_feature_names (list[str]): List of numerical features to be stored.
            max_depth (int): Maximum tree depth for the incremental decision trees. Defaults to 5.
            leaf_reservoir_length (int): Size of the reservoir stored at each leaf node of each feature's incremental
                decision tree. Defaults to 10.
            grace_period (int): Grace period of the underlying river Hoeffding Adaptive Trees. Defaults to 200.
            seed (int, optional): Random seed of the underlying river Hoeffding Adaptive Trees. Defaults to None.
        """
        self.feature_names = cat_feature_names + num_feature_names
        self.cat_feature_names = cat_feature_names
        self.num_feature_names = num_feature_names
        self._leaf_reservoir_length = leaf_reservoir_length
        self._seen_samples = 0

        self._storage_x = {cat_feature: HoeffdingAdaptiveTreeClassifier(
            max_depth=max_depth, leaf_prediction='nba', binary_split=True,
            grace_period=grace_period, seed=seed)
            for cat_feature in self.cat_feature_names}
        self._storage_x.update(
            {num_feature: HoeffdingAdaptiveTreeRegressor(
                max_depth=max_depth, leaf_prediction='adaptive', binary_split=True,
                grace_period=grace_period, seed=seed)
                for num_feature in self.num_feature_names})

        self.performances = {num_feature: Rolling(R2(), window_size=1000) for num_feature in self.num_feature_names}
        self.performances.update(
            {cat_feature: Rolling(Accuracy(), window_size=1000) for cat_feature in self.cat_feature_names})

        self.data_reservoirs = {feature: {} for feature in self.feature_names}

    def update(self, x: Dict, y: Optional[Any] = None):
        """Given a data point, it updates the storage.

        Args:
            x: Features as List of Dicts
            y: Target as float or integer (not used)

        Returns:
            None
        """
        for feature_name in x.keys():
            if feature_name in self.feature_names:
                feature_model = self._storage_x[feature_name]
                x_i = {**x}
                y_i = x_i.pop(feature_name)
                feature_model.learn_one(x_i, y_i)
                self._update_data_reservoirs(feature_name, x_i, x)
                pred_i = feature_model.predict_one(x_i)
                self.performances[feature_name].update(y_i, pred_i)
        self._seen_samples += 1

    @staticmethod
    def get_path_through_tree(node, x_i) -> str:
        """Given a data point and a starting node, traverses the decision tree.

        Args:
            node: Root node of the model.
            x_i: Data point to traverse the tree with.

        Returns:
            str: The walked path through the decision tree.
        """
        walked_path = ''
        path = iter(walk_through_tree(node, x_i, until_leaf=True))
        for stop in path:
            walked_path += str(stop)
            if hasattr(stop, 'repr_split'):
                walked_path += "|" + str(stop.repr_split) + "|" + str(stop.branch_no(x_i))
            walked_path += NODE_SEPERATOR
        return walked_path

    def _update_data_reservoirs(self, feature_name, x_i, x):
        """Update the data reservoir at a leaf node with a new sample.

        Args:
            feature_name str: The feature for which to update the stored data reservoirs.
            x_i dict: The data point to find the leaf node with.
            x dict: The data point to be inserted in the leaf node's reservoir.
        """
        root_node = self._storage_x[feature_name]._root
        data_reservoir = self.data_reservoirs[feature_name]
        leaf_id = self.get_path_through_tree(root_node, x_i)
        if leaf_id not in data_reservoir:
            data_reservoir[leaf_id] = GeometricReservoirStorage(
                size=self._leaf_reservoir_length, store_targets=False, constant_probability=1.0)
            self._delete_outdated_reservoirs(feature_name, root_node)
        data_reservoir[leaf_id].update(x)

    def __call__(self, feature_name: Any) -> Tuple[Union[HoeffdingTreeRegressor, HoeffdingTreeClassifier], str]:
        """Given a feature name, returns the associated data reservoirs.

        Args:
            feature_name (str): The feature name for which to return the data reservoirs.

        Returns:
            (dict, str): Tuple of data reservoir and flag if it is stored as a numerical feature or categorical.

        Raises:
            ValueError: If `feature_name` is not stored as a categorical feature nor a numerical feature.
        """
        if feature_name in self.cat_feature_names:
            return self._storage_x[feature_name], 'cat'
        elif feature_name in self.num_feature_names:
            return self._storage_x[feature_name], 'num'
        else:
            raise ValueError(f"The {feature_name} is not stored.")

    def _delete_outdated_reservoirs(self, feature_name: str, root_node: typing.Union["Branch", "Leaf"]):
        """Deletes the outdated reservoirs that no longer are part of all paths in the decision trees.

        Args:
            feature_name (str): The feature name for which outdated leafs might be deleted.
            root_node ("Branch", "Leaf"): The root node of the feature decision tree.
        """
        all_leafs = get_all_tree_paths(root_node)
        reservoirs_labels = list(self.data_reservoirs[feature_name].keys())
        for reservoirs_label in reservoirs_labels:
            if reservoirs_label not in all_leafs:
                del self.data_reservoirs[feature_name][reservoirs_label]

    def __len__(self):
        """Returns the number of samples observed in the storage object."""
        return self._seen_samples
