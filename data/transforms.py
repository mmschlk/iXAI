import numpy as np

__all__ = [
    "DropNA",
]


class DropNA:
    """
    reference: https://stackoverflow.com/questions/25539311
    """

    def fit(self, X, y):
        return self

    @staticmethod
    def fit_transform(X, y):
        mask = (np.isnan(X).any(-1) | np.isnan(y))
        if hasattr(X, 'loc'):
            X = X.loc[~mask]
        else:
            X = X[~mask]
        if hasattr(y, 'loc'):
            y = y.loc[~mask]
        else:
            y = y[~mask]
        return X, y

    @staticmethod
    def transform(X):
        mask = np.isnan(X).any(-1)
        if hasattr(X, 'loc'):
            X = X.loc[~mask]
        else:
            X = X[~mask]
        return X
