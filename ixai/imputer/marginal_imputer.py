from .base import BaseImputer
import random


class MarginalImputer(BaseImputer):
    def __init__(self, model_function, sampling_strategy, storage_object):
        self.sampling_strategy = sampling_strategy
        self.storage_object = storage_object
        super().__init__(
            model_function=model_function
        )
        # TODO - random seed - create separate issue

    def _sample(self, storage_object, feature_subset):
        features, _ = storage_object.get_data()
        if self.sampling_strategy == 'joint':
            sampled_features = self._sample_marginals(features, feature_subset)
        else:
            sampled_features = self._sample_product_marginals(features, feature_subset)
        return sampled_features

    def impute(self, feature_subset, x_i, n_samples=1):
        predictions = []
        for _ in range(n_samples):
            sampled_values = self._sample(self.storage_object, feature_subset)
            prediction = self.model_function({**x_i, **sampled_values})
            predictions.append(prediction)
        return predictions

    @staticmethod
    def _sample_marginals(features, feature_subset):
        rand_idx = random.randrange(len(features))
        sampled_instance = features[rand_idx].copy()
        sampled_features = {feature_name: sampled_instance[feature_name]
                            for feature_name in feature_subset}
        return sampled_features

    @staticmethod
    def _sample_product_marginals(features, feature_subset):
        sampled_features = {}
        for feature_name in feature_subset:
            rand_idx = random.randrange(len(features))
            sampled_features[feature_name] = features[
                            rand_idx].copy()[feature_name]
        return sampled_features
