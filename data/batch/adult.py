import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from data.batch._base import BaseBatchDataset, get_open_ml_dataset


class Adult(BaseBatchDataset):

    def __init__(
            self,
            version=2,
            random_seed=None,
            shuffle_dataset=False,
            n_samples=None
    ):
        assert version in [1, 2], "OpenML census dataset version must be '1' or '2'."
        dataset, class_label = get_open_ml_dataset("adult", version=version)
        self.num_feature_names = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt']
        self.cat_feature_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'education-num']
        dataset[self.num_feature_names] = dataset[self.num_feature_names].apply(pd.to_numeric)
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler())
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
        super(Adult, self).__init__(
            dataset=dataset,
            class_label=class_label,
            classification=True,
            random_seed=random_seed,
            shuffle_dataset=shuffle_dataset,
            n_samples=n_samples
        )


if __name__ == "__main__":
    test_dataset = Adult(random_seed=42, shuffle_dataset=False)
    print(f"n_samples:     {test_dataset.n_samples}")
    print(f"n_features:    {test_dataset.n_features}")
    print(f"feature_names: {test_dataset.feature_names}")
    stream = test_dataset.stream
    for i, sample in enumerate(stream):
        print(i, sample)
        if i > 3:
            break