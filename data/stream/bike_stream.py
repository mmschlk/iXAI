from typing import Optional

from river.datasets import Bikes as RiverDataset
from data.stream._base import StreamDataset


class Bike(StreamDataset):

    def __init__(
            self,
            n_samples: Optional[int] = 182470
    ):
        if n_samples is None:
            n_samples = 182470
        assert n_samples <= 182470, f"Bike contains at maximum  182470 samples."
        stream = RiverDataset()
        feature_names = ['moment', 'station', 'clouds', 'description', 'humidity', 'pressure', 'temperature', 'wind']
        cat_feature_names = ['station', 'description']
        num_feature_names = ['clouds', 'humidity', 'pressure', 'temperature', 'wind']
        super().__init__(
            stream=stream,
            n_samples=n_samples,
            feature_names=feature_names,
            cat_feature_names=cat_feature_names,
            num_feature_names=num_feature_names,
            task=stream.task,
            n_features=len(feature_names),
            n_outputs=stream.n_outputs,
        )


if __name__ == "__main__":
    dataset = Bike()
    stream = dataset.stream
    print(stream.n_samples)
    for n, (x_i, y_i) in enumerate(stream):
        print(n, x_i, y_i)
        if n > 3:
            print("\n", dataset.feature_names, "\n", list(x_i.keys()))
            break
