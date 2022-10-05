from .base_imputer import BaseImputer
import random


class MarginalImputer(BaseImputer):
    def __init__(self, sampling_strategy):
        self.sampling_strategy = sampling_strategy
        # TODO - random seed - create separate issue

    def _sample(self, storage_object, feature_subset):
        features, _ = storage_object.get_data()
        if self.sampling_strategy == 'joint':
            rand_idx = random.randrange(len(features))
            sampled_instance = features[rand_idx].copy()
            sampled_features = {feature_name: sampled_instance[feature_name]
                                for feature_name in feature_subset}
        else:
            sampled_features = {}
            for feature_name in feature_subset:
                rand_idx = random.randrange(len(features))
                sampled_features[feature_name] = features[
                                rand_idx].copy()[feature_name]
        return sampled_features

    def impute(self, model, storage_object, n_samples,
               feature_subset, x_i):
        predictions = []
        for _ in range(n_samples):
            sampled_values = self._sample(storage_object, feature_subset)
            new_x_i = x_i.copy()
            for key in feature_subset:
                new_x_i[key] = sampled_values[key]
            prediction = model.predict_one(new_x_i)
            predictions.append(prediction)
        return predictions
