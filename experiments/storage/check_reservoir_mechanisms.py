from increment_explain.storage import UniformReservoirStorage, GeometricReservoirStorage
import matplotlib.pyplot as plt


def dummy_stream(t):
    return [({'t': i}, i) for i in range(t)]


if __name__ == "__main__":

    STORE_TARGETS = False
    SIZE = 100
    T = 1000

    reservoir_values = {i: 0 for i in range(T)}
    stream = dummy_stream(t=T)
    for _ in range(1000):
        storage = UniformReservoirStorage(store_targets=STORE_TARGETS, size=SIZE)
        for x_i, y_i in stream:
            storage.update(x_i, y_i)
        for feature in storage.get_data()[0]:
            reservoir_values[feature['t']] += 1

    plt.bar(reservoir_values.keys(), reservoir_values.values())
    plt.title(f"Uniform Reservoir: size {SIZE}, T {T}")
    plt.show()

    reservoir_values = {i: 0 for i in range(T)}
    stream = dummy_stream(t=T)
    for _ in range(1000):
        storage = GeometricReservoirStorage(store_targets=STORE_TARGETS, size=SIZE, constant_probability=1)
        for x_i, y_i in stream:
            storage.update(x_i, y_i)
        for feature in storage.get_data()[0]:
            reservoir_values[feature['t']] += 1

    plt.bar(reservoir_values.keys(), reservoir_values.values())
    plt.title(f"Geometric Reservoir: size {SIZE}, T {T}")
    plt.show()
