import math
from typing import Callable
from ixai.storage.tree_storage import TreeStorage
from .base import BaseImputer
from river.base import Classifier, Regressor
from river.tree import HoeffdingTreeRegressor
import numpy as np
import random

EPS = 0.0000001


class TreeImputer(BaseImputer):
    def __init__(
            self,
            model_function: Callable,
            storage_object: TreeStorage,
            direct_predict_numeric: bool = False,
            use_storage: bool = False
    ):
        self.storage_object = storage_object
        self.direct_predict_numeric = direct_predict_numeric
        self.use_storage = use_storage
        super().__init__(
            model_function=model_function
        )

    def _sample_cat_feature(self, feature_name: str, feature_model: Classifier, x_i: dict, n_samples: int = 1):
        x_i_pred = {**x_i}
        _ = x_i_pred.pop(feature_name)
        probas = feature_model.predict_proba_one(x_i_pred)
        feature_values = list(probas.keys())
        feature_weights = list(probas.values())
        feature_value = random.choices(population=feature_values, weights=feature_weights, k=n_samples)[0]
        return feature_value

    def _sample_num_feature(
            self,
            feature_name: str,
            feature_model: HoeffdingTreeRegressor,
            x_i: dict,
            n_samples: int = 1
    ):
        x_i_pred = {**x_i}
        _ = x_i_pred.pop(feature_name)
        if not self.direct_predict_numeric:
            if hasattr(feature_model._root, "traverse"):
                leaf_nodes = feature_model._root.traverse(x_i_pred)
            else:
                leaf_nodes = [feature_model._root]
            leaf_node = leaf_nodes[0]
            if len(leaf_nodes) > 1:
                weights = [leaf_node.stats.n for leaf_node in leaf_nodes]
                leaf_node = random.choices(leaf_nodes, weights=weights)[0]
            mean = leaf_node.stats.mean.get()
            std = math.sqrt(abs(leaf_node.stats.get()))
            feature_value = np.random.normal(loc=mean, scale=std)
        else:
            feature_value = feature_model.predict_one(x_i_pred)
        return feature_value

    def _sample(self, feature_name, x_i, n_samples: int = 1):
        feature_model, feature_type = self.storage_object(feature_name)
        if feature_type == 'cat':
            feature_value = self._sample_cat_feature(feature_name, feature_model, x_i, n_samples=n_samples)
        else:  # feature_type == 'num'
            feature_value = self._sample_num_feature(feature_name, feature_model, x_i, n_samples=n_samples)
        return feature_value

    def _sample_from_storages(self, feature_name, x_i, n_samples: int = 1):
        feature_model, feature_type = self.storage_object(feature_name)
        data_reservoir = self.storage_object.data_reservoirs[feature_name]
        leaf_id = self.storage_object.get_path_through_tree(feature_model._root, x_i)
        try:
            storage = data_reservoir[leaf_id]
            x_storage, _ = storage.get_data()
            random_index = random.randint(0, len(x_storage) - 1)
            x_sampled = x_storage[random_index]
            sampled_feature_value = x_sampled[feature_name]
        except KeyError:
            sampled_feature_value = self._sample(feature_name=feature_name, x_i=x_i)
        return sampled_feature_value

    def impute(self, feature_subset, x_i, n_samples: int = 1):
        predictions = []
        for _ in range(n_samples):
            sampled_values = {}
            for feature_name in feature_subset:
                if self.use_storage:
                    sampled_value = self._sample_from_storages(feature_name, x_i, n_samples=n_samples)
                else:
                    sampled_value = self._sample(feature_name, x_i, n_samples=n_samples)
                sampled_values[feature_name] = sampled_value
            prediction = self.model_function({**x_i, **sampled_values})
            predictions.append(prediction)
        return predictions
