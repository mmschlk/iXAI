API Reference
=============
This page contains the API reference for public objects and functions in ``ixai``.


.. autosummary::
    :toctree: api
    :recursive:

    explainer
    storage
    imputer
    utils

Explainer
---------
.. autosummary::
    :nosignatures:

    explainer.IncrementalPFI
    explainer.sage.IncrementalSage
    explainer.sage.BatchSage
    explainer.sage.IntervalSage

Storage
-------
.. autosummary::
    :nosignatures:

    storage.GeometricReservoirStorage
    storage.UniformReservoirStorage
    storage.TreeStorage
    storage.IntervalStorage
    storage.SequenceStorage
    storage.BatchStorage

Imputer
-------
.. autosummary::
    :nosignatures:

    imputer.DefaultImputer
    imputer.MarginalImputer
    imputer.TreeImputer

Utils
-----
.. autosummary::
    :nosignatures:

    utils.wrappers
    utils.tracker
    utils.validators

Wrappers
~~~~~~~~
.. autosummary::
    :nosignatures:

    utils.wrappers.RiverWrapper
    utils.wrappers.TorchWrapper
    utils.wrappers.SklearnWrapper
    utils.wrappers.RiverMetricToLossFunction
    utils.wrappers.base.Wrapper

Tracker
~~~~~~~
.. autosummary::
    :nosignatures:

    utils.tracker.SlidingWindowTracker
    utils.tracker.ExponentialSmoothingTracker
    utils.tracker.WelfordTracker
    utils.tracker.MultiValueTracker
    utils.tracker.base.Tracker

Validators
~~~~~~~~~~
.. autosummary::
    :nosignatures:

    utils.validators.validate_loss_function
    utils.validators.validate_model_function