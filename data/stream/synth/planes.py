from typing import Optional
from river.datasets.synth import Planes2D as RiverDataset

from data.stream._base import StreamDataset

class Planes2D(StreamDataset):

    def __init__(
            self,
            random_seed: Optional[int] = None,
            n_samples:  Optional[int] = None
    ):
        if n_samples is None:
            n_samples = 20000
        stream = RiverDataset(seed=random_seed)
        feature_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        cat_feature_names = []
        num_feature_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
    dataset = Planes2D(random_seed=42)
    stream = dataset.stream
    print(stream.n_samples)
    for n, (x_i, y_i) in enumerate(stream):
        print(n, x_i, y_i)
        if n > 3:
            print("\n", dataset.feature_names, "\n", list(x_i.keys()))
            break
