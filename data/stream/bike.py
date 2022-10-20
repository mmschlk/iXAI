from river.datasets import Bikes as RiverDataset


class Elec2:

    def __init__(
            self,
    ):
        self.stream = RiverDataset()
        self.feature_names = ['moment', 'station', 'clouds', 'description', 'humidity', 'pressure', 'temperature', 'wind']
        self.cat_feature_names = ['station', 'description']
        self.num_feature_names = ['clouds', 'humidity', 'pressure', 'temperature', 'wind']
        self.n_samples = 182470


if __name__ == "__main__":
    dataset = Elec2()
    stream = dataset.stream
    print(stream.n_samples)
    for n, (x_i, y_i) in enumerate(stream):
        print(n, x_i, y_i)
        if n > 3:
            print("\n", dataset.feature_names, "\n", list(x_i.keys()))
            break
