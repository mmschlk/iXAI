from river.datasets import CreditCard as RiverDataset


class CreditCard:

    def __init__(
            self,
    ):
        self.stream = RiverDataset()
        self.feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        self.cat_feature_names = []
        self.num_feature_names = self.feature_names
        self.n_samples = 284_807


if __name__ == "__main__":
    dataset = CreditCard()
    stream = dataset.stream
    print(stream.n_samples)
    for n, (x_i, y_i) in enumerate(stream):
        print(n, x_i, y_i)
        if n > 3:
            print("\n", dataset.feature_names, "\n", list(x_i.keys()))
            break
