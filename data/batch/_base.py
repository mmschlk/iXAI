from abc import ABCMeta, abstractmethod
from typing import Optional, Union
import openml
from sklearn.utils import shuffle
from river.stream import iter_pandas
from river.datasets import base
from sklearn.preprocessing import LabelEncoder
import numpy as np

from data.stream import BatchStream, SuddenDriftStream, ConceptDriftStream


__all__ = [
    "get_open_ml_dataset",
    "BaseBatchDataset",
]


def get_open_ml_dataset(open_ml_id, version=1):
    dataset = openml.datasets.get_dataset(open_ml_id, version=version, download_data=True)
    class_label = dataset.default_target_attribute
    x_data = dataset.get_data()[0]
    return x_data, class_label


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
        self.feature_names = list(self.x_data.columns)
        self.n_features = len(self.feature_names)
        self.n_samples = len(self.x_data)
        self.classification = classification
        if self.classification:
            self.task = base.BINARY_CLF
            self._label_encoder = LabelEncoder()
            self._transform_label_column()
        else:
            self.task = base.REG
        self.n_outputs = 1

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
            concept_drift_stream = ConceptDriftStream(stream=stream_1, drift_stream=stream_2,
                                                      width=width)
        else:
            concept_drift_stream = SuddenDriftStream(stream=stream_1, drift_stream=stream_2,
                                                     task=self.task, n_features=self.n_features,
                                                     n_outputs=self.n_outputs)
        return concept_drift_stream

    def transform_features(self, feature_names, transformer):
        for feature in feature_names:
            self.x_data[feature] = transformer.fit_transform(self.x_data[feature].values.reshape(-1, 1))

    def _transform_label_column(self):
        self.y_data[:] = self._label_encoder.fit_transform(self.y_data)

    @property
    def label_encoding(self):
        return self._label_encoder.get_params()


if __name__ == "__main__":
    pass
