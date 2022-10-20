from typing import Optional

from river.datasets.synth import Planes2D as RiverDataset


class Planes2D:

    def __init__(
            self,
            random_seed: Optional[int] = None
    ):
        self.stream = RiverDataset(seed=random_seed)
        self.feature_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.cat_feature_names = []
        self.num_feature_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.n_samples = None


if __name__ == "__main__":
    dataset = Planes2D(random_seed=42)
    stream = dataset.stream
    print(stream.n_samples)
    for n, (x_i, y_i) in enumerate(stream):
        print(n, x_i, y_i)
        if n > 30:
            print("\n", dataset.feature_names, "\n", list(x_i.keys()))
            break
