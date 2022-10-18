from .base_imputer import BaseImputer


class DefaultImputer(BaseImputer):
    def __init__(self, model_function, values):
        self.values = values
        super().__init__(
            model_function=model_function
        )

    def impute(self, feature_subset, x_i, n_samples=None):
        new_x_i = x_i.copy()
        for key in feature_subset:
            new_x_i[key] = self.values[key]
        prediction = self.model_function(new_x_i)
        return [prediction]
