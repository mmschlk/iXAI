from typing import Optional
from river.datasets.synth import Agrawal as RiverDataset

from data.stream._base import StreamDataset


class Agrawal(StreamDataset):

    def __init__(
            self,
            classification_function,
            random_seed: Optional[int] = None,
            n_samples: Optional[int] = None
    ):
        if n_samples is None:
            n_samples = 20000
        stream = RiverDataset(classification_function=classification_function, seed=random_seed)
        feature_names = ['salary', 'commission', 'age', 'elevel', 'car', 'zipcode', 'hvalue', 'hyears', 'loan']
        cat_feature_names = ['elevel', 'car', 'zipcode']
        num_feature_names = ['salary', 'commission', 'age', 'hvalue', 'hyears', 'loan']
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
    dataset = Agrawal(classification_function=1, random_seed=42)
    stream = dataset.stream
    print(stream.n_samples)
    for n, (x_i, y_i) in enumerate(stream):
        print(n, x_i, y_i)
        if n > 3:
            print("\n", dataset.feature_names, "\n", list(x_i.keys()))
            break
