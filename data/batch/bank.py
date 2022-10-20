import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from data.batch._base import BaseBatchDataset, get_open_ml_dataset


OPEN_ML_BANK_MARKETING_RENAME_MAPPER = {
    'V1': 'age',
    'V2': 'job',
    'V3': 'marital',
    'V4': 'education',
    'V5': 'default',
    'V6': 'balance',
    'V7': 'housing',
    'V8': 'loan',
    'V9': 'contact',
    'V10': 'day',
    'V11': 'month',
    'V12': 'duration',
    'V13': 'campaign',
    'V14': 'pdays',
    'V15': 'previous',
    'V16': 'poutcome'
}


class BankMarketing(BaseBatchDataset):

    def __init__(
            self,
            random_seed=None,
            shuffle_dataset=False
    ):
        dataset, class_label = get_open_ml_dataset("bank-marketing", version=1)
        dataset = dataset.rename(columns=OPEN_ML_BANK_MARKETING_RENAME_MAPPER)
        self.num_feature_names = ['age', 'balance', 'duration', 'pdays', 'previous']
        self.cat_feature_names = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'campaign', 'poutcome']
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
        super(BankMarketing, self).__init__(
            dataset=dataset,
            class_label=class_label,
            classification=True,
            random_seed=random_seed,
            shuffle_dataset=shuffle_dataset,
        )


if __name__ == "__main__":
    test_dataset = BankMarketing(random_seed=42, shuffle_dataset=False)
    print(f"n_samples:     {test_dataset.n_samples}")
    print(f"n_features:    {test_dataset.n_features}")
    print(f"feature_names: {test_dataset.feature_names}")
    stream = test_dataset.stream
    for i, sample in enumerate(stream):
        print(i, sample)
        if i > 3:
            break