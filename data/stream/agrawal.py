from typing import Optional

from river.datasets.synth import Agrawal as RiverDataset


class Agrawal:

    def __init__(
            self,
            classification_function,
            random_seed: Optional[int] = None
    ):
        self.stream = RiverDataset(classification_function=classification_function, seed=random_seed)
        self.feature_names = ['salary', 'commission', 'age', 'elevel', 'car', 'zipcode', 'hvalue', 'hyears', 'loan']
        self.cat_feature_names = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'campaign', 'poutcome']
        self.num_feature_names = ['elevel', 'car', 'zipcode']
        self.n_samples = None


if __name__ == "__main__":
    dataset = Agrawal(classification_function=1, random_seed=42)
    stream = dataset.stream
    print(stream.n_samples)
    for n, (x_i, y_i) in enumerate(stream):
        print(n, x_i, y_i)
        if n > 3:
            print("\n", dataset.feature_names, "\n", list(x_i.keys()))
            break
