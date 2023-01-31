"""ixai is a library for Incremental Explainable Artificial Intelligence. It provides specific XAI
explanation methods that naturally work with dynamic time-dependent models. Commonly such models are
 trained in the incremental learning paradigm. Such models can easily be explained with ixai.
"""
import warnings

from .__version__ import __version__

# incremental explainers
from .explainer import IncrementalSage
from .explainer import IncrementalPFI
# batch-based explainers
from .explainer import IntervalSage
from .explainer import BatchSage

# incremental storages
from .storage import UniformReservoirStorage
from .storage import GeometricReservoirStorage
from .storage import IntervalStorage
from .storage import SequenceStorage
from .storage import TreeStorage
from .storage import BatchStorage

# imputer
from .imputer import DefaultImputer
from .imputer import MarginalImputer
from .imputer import TreeImputer

# utils.wrapper
from .utils.wrappers import SklearnWrapper
from .utils.wrappers import TorchWrapper
from .utils.wrappers import RiverWrapper

_no_visualization_warning = "Visualization is not available, because matplotlib is not installed." \
                            " Run `pip install matplotlib` to fix this."


def unsupported_function(*args, **kwargs):
    warnings.warn(_no_visualization_warning)


class UnsupportedModule(object):
    def __getattribute__(self, item):
        raise ValueError(_no_visualization_warning)


try:
    import matplotlib
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
if matplotlib_available:
    from .visualization import FeatureImportancePlotter
    from .visualization import ChangePlotter
else:
    FeatureImportancePlotter = unsupported_function
    ChangePlotter = unsupported_function
    visualization = UnsupportedModule()


__all__ = [
    # explainers
    "IncrementalSage",
    "IntervalSage",
    "BatchSage",
    "IncrementalPFI",
    # storage
    "UniformReservoirStorage",
    "GeometricReservoirStorage",
    "IntervalStorage",
    "SequenceStorage",
    "TreeStorage",
    "BatchStorage",
    # imputer
    "DefaultImputer",
    "MarginalImputer",
    "TreeImputer",
    # wrapper
    "RiverWrapper",
    "SklearnWrapper",
    "TorchWrapper",
    # visualization
    "FeatureImportancePlotter",
    "ChangePlotter",
]
