from river.datasets import Elec2 as RiverDataset


class Elec2:

    def __init__(
            self,
    ):
        self.stream = RiverDataset()
        self.feature_names = ['date', 'day', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']
        self.cat_feature_names = []
        self.num_feature_names = self.feature_names
        self.n_samples = 45312


if __name__ == "__main__":
    dataset = Elec2()
    stream = dataset.stream
    print(stream.n_samples)
    for n, (x_i, y_i) in enumerate(stream):
        print(n, x_i, y_i)
        if n > 3:
            print("\n", dataset.feature_names, "\n", list(x_i.keys()))
            break
