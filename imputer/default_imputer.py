from .base_imputer import BaseImputer


class DefaultImputer(BaseImputer):
    def __init__(self, values):
        self.values = values

    def impute(self, model, feature_subset, x_i, values=None):
        if values is None:
            values = self.values
        new_x_i = x_i.copy()
        for key in feature_subset:
            new_x_i[key] = values[key]
        prediction = model.predict_one(new_x_i)
        return [prediction]
