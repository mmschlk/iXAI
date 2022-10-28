from random import random
from typing import Optional, Generator, Union, Dict
import numpy as np
from river.datasets.base import Dataset


__all__ = [
    "BatchStream",
    "FeatureSwitchStream",
    "ConceptDriftStream",
    "StreamDataset"
]


class BatchStream(Dataset):

    def __init__(self, stream, task, n_features, n_classes=None, n_outputs=None):
        super().__init__(task, n_features, n_classes=n_classes, n_outputs=n_outputs)
        self.stream_gen = stream

    def __iter__(self):
        for x_i, y_i in self.stream_gen:
            yield x_i, y_i


class ConceptDriftStream(Dataset):

    def __init__(
            self,
            stream,
            drift_stream,
            position: int = 5000,
            width: int = 1000,
            alpha: float = None,
    ):
        super().__init__(
            n_features=stream.n_features,
            n_classes=stream.n_classes,
            n_outputs=stream.n_outputs,
            task=stream.task,
        )
        self.alpha = alpha
        if self.alpha is not None:
            if 0 < self.alpha <= 90.0:
                w = int(1 / np.tan(self.alpha * np.pi / 180))
                self.width = w if w > 0 else 1
            else:
                raise ValueError(
                    f"Invalid alpha value: {alpha}. " f"Valid values are in the range (0.0, 90.0]"
                )
        else:
            self.width = width
        self.position = position
        self.stream = stream
        self.drift_stream = drift_stream

    def __iter__(self):
        stream_generator = iter(self.stream)
        drift_stream_generator = iter(self.drift_stream)
        sample_idx = 0

        while True:
            sample_idx += 1
            v = -4.0 * float(sample_idx - self.position) / float(self.width)
            probability_drift = 1.0 / (1.0 + np.exp(v))
            try:
                if random() > probability_drift:
                    try:
                        x, y = next(stream_generator)
                    except StopIteration:
                        x, y = next(drift_stream_generator)
                else:
                    try:
                        x, y = next(drift_stream_generator)
                    except StopIteration:
                        x, y = next(stream_generator)
            except StopIteration:
                break
            yield x, y


class FeatureSwitchStream(Dataset):

    def __init__(self, feature_switch, stream, task, n_features, n_classes=None, n_outputs=None):
        super().__init__(task, n_features, n_classes=n_classes, n_outputs=n_outputs)
        self.stream = stream
        self.feature_switch = feature_switch

    def __iter__(self):
        for x, y in self.stream.__iter__():
            for feature_1, feature_2 in self.feature_switch.items():
                x[feature_1], x[feature_2] = x[feature_2], x[feature_1]
            yield x, y


class StreamDataset(Dataset):

    def __init__(self, stream, n_samples, task, n_features, n_outputs, feature_names, cat_feature_names, num_feature_names):
        super().__init__(task, n_features)
        self.stream = stream
        self.n_outputs = n_outputs
        self.feature_names = feature_names
        self.cat_feature_names = cat_feature_names
        self.num_feature_names = num_feature_names
        self.n_samples = n_samples

    def __iter__(self):
        n = 0
        stream_iter = iter(self.stream)
        while n < self.n_samples:
            yield next(stream_iter)
            n += 1


def slice_stream(stream: Generator, position: int):
    first_stream = []
    second_stream = []
    for n, stream_i in enumerate(stream):
        if n < position:
            first_stream.append(stream_i)
        else:
            second_stream.append(stream_i)
    first_stream = BatchStream(
        stream=first_stream,
        task=stream.task, n_features=stream.n_features, n_outputs=stream.n_outputs, n_classes=stream.n_classes)
    second_stream = BatchStream(
        stream=second_stream,
        task=stream.task, n_features=stream.n_features, n_outputs=stream.n_outputs, n_classes=stream.n_classes)
    return first_stream, second_stream


def _get_feature_remapping_dict(feature_remapping_str: str):
    feature_remapping = {}
    for features in feature_remapping_str.split(' '):
        feature_1, feature_2 = features.split('_')
        feature_remapping[feature_1] = feature_2
        feature_remapping[feature_2] = feature_1
    return feature_remapping


def get_concept_drift_stream(
        stream,
        position: int,
        drift_stream: Optional[Generator] = None,
        feature_remapping: Optional[Union[Dict[str, str], str]] = None,
        width: int = 1
):

    if drift_stream is None:
        stream, drift_stream = slice_stream(stream=stream, position=position)

    if feature_remapping is not None:
        if type(feature_remapping) == str:
            feature_remapping = _get_feature_remapping_dict(feature_remapping)
        drift_stream = FeatureSwitchStream(
            feature_switch=feature_remapping, stream=drift_stream,
            task=drift_stream.task, n_features=drift_stream.n_features,
            n_classes=drift_stream.n_classes, n_outputs=drift_stream.n_outputs)

    concept_drift_stream = ConceptDriftStream(
        stream=stream, drift_stream=drift_stream, position=position, width=width)

    return concept_drift_stream


if __name__ == "__main__":
    pass
