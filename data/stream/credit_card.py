from typing import Optional

from river.datasets import CreditCard as RiverDataset
from data.stream._base import StreamDataset


class CreditCard(StreamDataset):

    def __init__(
            self,
            n_samples: Optional[int] = 284_807
    ):
        if n_samples is None:
            n_samples = 284_807
        assert n_samples <= 284_807, f"CreditCard contains at maximum  284_807 samples."
        stream = RiverDataset()
        feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        cat_feature_names = []
        num_feature_names = feature_names
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
    dataset = CreditCard()
    stream = dataset.stream
    print(stream.n_samples)
    for n, (x_i, y_i) in enumerate(stream):
        print(n, x_i, y_i)
        if n > 3:
            print("\n", dataset.feature_names, "\n", list(x_i.keys()))
            break
