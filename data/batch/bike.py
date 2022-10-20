import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from data.batch._base import BaseBatchDataset, get_open_ml_dataset


class BikeSharing(BaseBatchDataset):

    def __init__(
            self,
            random_seed=None,
            shuffle_dataset=False
    ):
        dataset, class_label = get_open_ml_dataset(42713, version=1)
        self.num_feature_names = ['hour', 'temp', 'feel_temp', 'humidity', 'windspeed']
        self.cat_feature_names = ['season', 'year', 'month', 'holiday', 'weekday', 'workingday', 'weather']
        dataset[self.num_feature_names] = dataset[self.num_feature_names].apply(pd.to_numeric)
        num_pipeline = Pipeline([
            ('scaler', RobustScaler())
        ])
        cat_pipeline = Pipeline([
            ('ordinal_encoder', OrdinalEncoder()),
        ])
        column_transformer = ColumnTransformer([
            ('numerical', num_pipeline, self.num_feature_names),
            ('categorical', cat_pipeline, self.cat_feature_names),
        ], remainder='passthrough')
        col_names = self.num_feature_names + self.cat_feature_names
        col_names += [feature for feature in dataset.columns if feature not in col_names]
        dataset = pd.DataFrame(column_transformer.fit_transform(dataset), columns=col_names)
        dataset.dropna(inplace=True)
        super(BikeSharing, self).__init__(
            dataset=dataset,
            class_label=class_label,
            classification=False,
            random_seed=random_seed,
            shuffle_dataset=shuffle_dataset,
        )


if __name__ == "__main__":
    test_dataset = BikeSharing(random_seed=42, shuffle_dataset=False)
    print(f"n_samples:     {test_dataset.n_samples}")
    print(f"n_features:    {test_dataset.n_features}")
    print(f"feature_names: {test_dataset.feature_names}")
    stream = test_dataset.stream
    for i, sample in enumerate(stream):
        print(i, sample)
        if i > 3:
            break
