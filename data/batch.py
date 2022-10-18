from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Any

import openml
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

from river.datasets.base import Dataset
from river.datasets.synth import ConceptDriftStream
from river.stream import iter_pandas
from river.datasets import base

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from sklearn.compose import ColumnTransformer

import sklearn
import river
import numpy as np


from data.stream import BatchStream, SuddenDriftStream


__all__ = [
    "Adult",
    "BankMarketing",
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
    x_data = dataset.get_data()[0]
    return x_data, class_label


# =============================================================================
# Base Datasets
# =============================================================================


class BaseBatchDataset(metaclass=ABCMeta):
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
            classification: bool,
            random_seed: Optional[int] = None,
            drop_na: bool = True,
            shuffle_dataset: bool = True,
    ):
        if drop_na:
            dataset.dropna(inplace=True)
        if shuffle_dataset:
            dataset = shuffle(dataset, random_state=random_seed)
        self.x_data = dataset
        self.y_data = dataset.pop(class_label)
        self.feature_names = self.x_data.columns
        self.n_features = len(self.feature_names)
        self.n_samples = len(self.x_data)
        self.classification = classification
        if self.classification:
            self.task = base.BINARY_CLF
            self._label_encoder = LabelEncoder()
        else:
            self.task = base.REG
            self._label_encoder = MinMaxScaler(feature_range=(0, 1))
        self.n_outputs = 1
        self._transform_label_column()

    def to_stream(self):
        return BatchStream(
            stream=iter_pandas(X=self.x_data, y=self.y_data),
            task=self.task, n_features=self.n_features, n_outputs=self.n_outputs
        )

    def to_concept_drift_stream(
            self,
            feature_remapping: dict[str, str],
            position: Union[int, float] = 0.5,
            width: Union[int, float] = 0.05,
            sudden_drift: bool = False
    ):
        feature_remapping.update({v: k for k, v in feature_remapping.items()})
        n_stream_1 = int(np.floor(position * self.n_samples)) if position < 1 else int(position)
        n_stream_2 = self.n_samples - n_stream_1
        width = int(width * self.n_samples) if width < 1 else int(width)

        x_data_1, y_data_1 = self.x_data[:n_stream_1], self.y_data[:n_stream_1]
        x_data_2, y_data_2 = self.x_data[n_stream_1:], self.y_data[n_stream_1:]
        x_data_2 = x_data_2.rename(columns=feature_remapping, inplace=False)
        stream_1 = BatchStream(
            stream=iter_pandas(X=x_data_1, y=y_data_1),
            task=self.task, n_features=self.n_features, n_outputs=self.n_outputs
        )
        stream_2 = BatchStream(
            stream=iter_pandas(X=x_data_2, y=y_data_2),
            task=self.task, n_features=self.n_features, n_outputs=self.n_outputs
        )
        if not sudden_drift:
            concept_drift_stream = ConceptDriftStream(stream=stream_1, drift_stream=stream_2, width=width)
        else:
            concept_drift_stream = SuddenDriftStream(stream=stream_1, drift_stream=stream_2,
                                                     task=self.task, n_features=self.n_features, n_outputs=self.n_outputs)
        return concept_drift_stream

    def transform_features(self, feature_names, transformer):
        for feature in feature_names:
            self.x_data[feature] = transformer.fit_transform(self.x_data[feature].values.reshape(-1, 1))

    def _transform_label_column(self):
        self.y_data[:] = self._label_encoder.fit_transform(self.y_data)

    @property
    def label_encoding(self):
        return self._label_encoder.get_params()


# ======================================================================================================================
# Public Datasets
# ======================================================================================================================

# Adult (Census) Dataset -----------------------------------------------------------------------------------------------


class Adult(BaseBatchDataset):

    def __init__(
            self,
            version=2,
            random_seed=None,
            shuffle_dataset=True
    ):
        assert version in [1, 2], "OpenML census dataset version must be '1' or '2'."
        dataset, class_label = get_open_ml_dataset("adult", version=version)
        self.num_feature_names = OPEN_ML_ADULT_NUM_FEATURES
        self.cat_feature_names = OPEN_ML_ADULT_CATEGORICAL_FEATURE_NAMES
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
        dataset = pd.DataFrame(column_transformer.fit_transform(dataset), columns=dataset.columns)
        dataset.dropna(inplace=True)
        super(Adult, self).__init__(
            dataset=dataset,
            class_label=class_label,
            classification=True,
            random_seed=random_seed,
            shuffle_dataset=shuffle_dataset,
        )


# Bank Marketing Dataset -----------------------------------------------------------------------------------------------


class BankMarketing(BaseBatchDataset):

    def __init__(
            self,
            random_seed=None,
            shuffle_dataset=True
    ):
        dataset, class_label = get_open_ml_dataset("bank-marketing", version=1)
        dataset = dataset.rename(columns=OPEN_ML_BANK_MARKETING_RENAME_MAPPER)
        self.num_feature_names = OPEN_ML_BANK_MARKETING_NUM_FEATURES
        self.cat_feature_names = OPEN_ML_BANK_MARKETING_CATEGORICAL_FEATURE_NAMES
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
        dataset = pd.DataFrame(column_transformer.fit_transform(dataset), columns=dataset.columns)
        dataset.dropna(inplace=True)
        super(BankMarketing, self).__init__(
            dataset=dataset,
            class_label=class_label,
            classification=True,
            random_seed=random_seed,
            shuffle_dataset=shuffle_dataset,
        )


if __name__ == "__main__":
    test_dataset = Adult(random_seed=42)
    stream = test_dataset.to_concept_drift_stream({'age': 'workclass'}, sudden_drift=True, position=2)
    for i, sample in enumerate(stream):
        print(i, sample)
        if i >= 3:
            break


