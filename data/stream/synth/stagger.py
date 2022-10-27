from typing import Optional
from river.datasets.synth import STAGGER as RiverDataset

from data.stream._base import StreamDataset


class Stagger(StreamDataset):

    def __init__(
            self,
            classification_function,
            random_seed: Optional[int] = None,
            n_samples: int = 10000
    ):
        stream = RiverDataset(classification_function=classification_function, seed=random_seed)
        feature_names = ['size', 'color', 'shape']
        cat_feature_names = ['size', 'color', 'shape']
        num_feature_names = []
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
    dataset = Stagger(classification_function=1, random_seed=42)
    stream = dataset.stream
    print(stream.n_samples)
    for n, (x_i, y_i) in enumerate(stream):
        print(n, x_i, y_i)
        if n > 3:
            print("\n", dataset.feature_names, "\n", list(x_i.keys()))
            break
