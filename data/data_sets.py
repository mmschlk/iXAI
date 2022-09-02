from abc import ABCMeta, abstractmethod

import openml
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import sklearn
import river
import numpy as np


__all__ = [
    "Adult",
]

# =============================================================================
# Types and constants
# =============================================================================

# stagger data stream
STAGGER_FEATURE_NAMES = np.array(
    [
        'size',
        'color',
        'shape'
    ]
)
STAGGER_CATEGORICAL_FEATURE_NAMES = np.array(
    [
        'size',
        'color',
        'shape'
    ]
)
STAGGER_LENGTH = 10000

# agrawal data stream
AGRAWAL_FEATURE_NAMES = np.array(
    [
        'salary',
        'commission',
        'age',
        'elevel',
        'car',
        'zipcode',
        'hvalue',
        'hyears',
        'loan'
    ]
)
AGRAWAL_CATEGORICAL_FEATURE_NAMES = np.array(
    [
        'elevel',
        'car',
        'zipcode'
    ]
)
AGRAWAL_LENGTH = 20000

# adult data stream (census)
OPEN_ML_ADULT_FEATURE_NAMES = np.array(
    [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capitalgain',
        'capitalloss',
        'hoursperweek',
        'native-country'
    ]
)
OPEN_ML_ADULT_CATEGORICAL_FEATURE_NAMES = np.array(
    [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country'
    ]
)
OPEN_ML_ADULT_NUM_FEATURES = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
OPEN_ML_ADULT_NAN_FEATURES = ['workclass', 'occupation', 'native-country']
OPEN_ML_ADULT_LENGTH = 45222

# bank-marketing data stream (bank)
OPEN_ML_BANK_MARKETING_FEATURE_NAMES = np.array(
    [
        'age',
        'job',
        'marital',
        'education',
        'default',
        'balance',
        'housing',
        'loan',
        'contact',
        'day',
        'month',
        'duration',
        'campaign',
        'pdays',
        'previous',
        'poutcome'
    ]
)
OPEN_ML_BANK_MARKETING_CATEGORICAL_FEATURE_NAMES = np.array(
    [
        'job',
        'marital',
        'education',
        'default',
        'housing',
        'loan',
        'contact',
        'day',
        'month',
        'campaign',
        'poutcome'
    ]
)
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
OPEN_ML_BANK_MARKETING_NUM_FEATURES = ['age', 'balance', 'duration', 'pdays', 'previous']
OPEN_ML_BANK_MARKETING_LENGTH = 45211

# bike data stream (bike rental regression)
BIKE_FEATURE_NAMES = np.array(
    [
        'season',
        'yr',
        'mnth',
        'hr',
        'holiday',
        'weekday',
        'workingday',
        'weathersit',
        'temp',
        'atemp',
        'hum',
        'windspeed'
    ]
)
BIKE_CATEGORICAL_FEATURE_NAMES = np.array(
    [
        'season',
        'yr',
        'mnth',
        'hr',
        'holiday',
        'weekday',
        'workingday',
        'weathersit'
    ]
)
BIKE_FEATURE_NUM_FEATURES = ['temp', 'atemp', 'hum', 'windspeed']

BIKE_LENGTH = 17379


def get_open_ml_dataset(open_ml_id, version=1):
    dataset = openml.datasets.get_dataset(open_ml_id, version=version, download_data=True)
    class_label = dataset.default_target_attribute
    print(f"Loaded openML dataset '{dataset.name}', the target feature is '{class_label}'.")
    x_data = dataset.get_data()[0]
    return x_data, class_label


# =============================================================================
# Base Datasets
# =============================================================================


class BaseDataset(metaclass=ABCMeta):
    """Base class for creating data sets.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(
            self,
            *,
            dataset,
            class_label,
            random_seed=None,
            drop_na=True,
            shuffle_dataset=True,
            feature_encoder=None,
            features_to_encode=(),
            feature_scaler=None,
            features_to_scale=()
    ):
        if drop_na:
            dataset.dropna(inplace=True)
        if shuffle_dataset:
            dataset = shuffle(dataset, random_state=random_seed)
        self.x_data = dataset
        self.y_data = dataset.pop(class_label)
        self.feature_names = self.x_data.columns
        if feature_encoder is not None:
            self.encode_features(features_to_encode=features_to_encode, feature_encoder=feature_encoder)
        if feature_scaler is not None:
            self.scale_features(features_to_scale=features_to_scale, feature_scaler=feature_scaler)
        self.n_features = len(self.feature_names)
        self.dataset_size = len(self.x_data)
        self._label_encoder = LabelEncoder()
        self._transform_label_column()

    def to_stream(self):
        return river.stream.iter_pandas(self.x_data, self.y_data)

    def _transform_features(self, feature_names, transformer):
        for feature in feature_names:
            self.x_data[feature] = transformer.fit_transform(self.x_data[feature].values.reshape(-1, 1))

    def scale_features(self, features_to_scale, feature_scaler: object):
        if feature_scaler == 'standard':
            feature_scaler = sklearn.preprocessing.StandardScaler()
        if feature_scaler == 'robust':
            feature_scaler = sklearn.preprocessing.Robustcaler()
        if not hasattr(feature_scaler, "fit_transform"):
            raise ValueError("Feature scaler must be either 'standard', 'robust', or a feature transformer instance "
                             "(like 'sklearn.preprocessing') implementing a 'fit_transform' method.")
        self._transform_features(feature_names=features_to_scale, transformer=feature_scaler)

    def encode_features(self, features_to_encode, feature_encoder):
        if feature_encoder == 'ordinal':
            feature_encoder = sklearn.preprocessing.OrdinalEncoder()
        if not hasattr(feature_encoder, "fit_transform"):
            raise ValueError("Feature encoder must be either 'ordinal', or a feature transformer instance "
                             "(like 'sklearn.preprocessing') implementing a 'fit_transform' method.")
        self._transform_features(feature_names=features_to_encode, transformer=feature_encoder)

    def _transform_label_column(self):
        self.y_data = self._label_encoder.fit_transform(y=self.y_data)

    @property
    def label_encoding(self):
        return self._label_encoder.get_params()


# =============================================================================
# Public Datasets
# =============================================================================


class Adult(BaseDataset):

    def __init__(
            self,
            version=2,
            random_seed=None,
            drop_na=True,
            shuffle_dataset=True,
            feature_encoder=None,
            features_to_encode=(),
            feature_scaler=None,
            features_to_scale=()
    ):
        assert version in [1, 2], "OpenML census dataset version must be '1' or '2'."
        dataset, class_label = get_open_ml_dataset("adult", version=version)
        dataset[OPEN_ML_ADULT_NUM_FEATURES] = dataset[OPEN_ML_ADULT_NUM_FEATURES].apply(pd.to_numeric)

        if feature_encoder is not None:
            if features_to_encode == ():
                features_to_encode = OPEN_ML_ADULT_CATEGORICAL_FEATURE_NAMES
        if feature_scaler is not None:
            if features_to_scale == ():
                features_to_scale = OPEN_ML_ADULT_NUM_FEATURES

        super(Adult, self).__init__(
            dataset=dataset,
            class_label=class_label,
            drop_na=drop_na,
            random_seed=random_seed,
            shuffle_dataset=shuffle_dataset,
            feature_encoder=feature_encoder,
            features_to_encode=features_to_encode,
            feature_scaler=feature_scaler,
            features_to_scale=features_to_scale
        )
        self.categorical_feature_names = OPEN_ML_ADULT_CATEGORICAL_FEATURE_NAMES
        self.classification = True


if __name__ == "__main__":
    test_dataset = Adult(random_seed=42, feature_encoder='ordinal', feature_scaler='standard')
    stream = test_dataset.to_stream()
    for i, sample in enumerate(stream):
        print(i, sample)
        if i >= 3:
            break
