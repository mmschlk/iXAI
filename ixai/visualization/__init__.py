try:
    import matplotlib
except ImportError:
    raise ImportError("Plotting is not available, because matplotlib is not installed. "
                      "Run `pip install matplotlib` to fix this.")

from .plotting import FeatureImportancePlotter
from .change_plotter import ChangePlotter

__all__ = [
    "FeatureImportancePlotter",
    "ChangePlotter"
]
