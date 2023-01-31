API Reference
=============
This page contains the API reference for public objects and functions in ixai.


.. explainer_api:

explainer
---------
.. autosummary::
   :toctree: generated

   ixai.explainer
   ixai.explainer.IncrementalPFI
   ixai.explainer.IncrementalSage
   ixai.explainer.IntervalSage
   ixai.explainer.BatchSage

.. imputer_api:

imputer
-------
.. autosummary::
   :toctree: generated

   ixai.imputer
   ixai.imputer.DefaultImputer
   ixai.imputer.MarginalImputer
   ixai.imputer.TreeImputer

.. storage_api:

storage
-------
.. autosummary::
   :toctree: generated

   ixai.storage
   ixai.storage.GeometricReservoirStorage
   ixai.storage.UniformReservoirStorage
   ixai.storage.TreeStorage
   ixai.storage.IntervalStorage
   ixai.storage.SequenceStorage
   ixai.storage.BatchStorage

utils
-----
.. autosummary::
   :toctree: generated

   ixai.utils.wrappers
   ixai.utils.tracker
   ixai.utils.validators

wrappers
~~~~~~~~
.. autosummary::
   :toctree: generated

   ixai.utils.wrappers.RiverWrapper
   ixai.utils.wrappers.TorchWrapper
   ixai.utils.wrappers.SklearnWrapper
   ixai.utils.wrappers.RiverMetricToLossFunction
   ixai.utils.wrappers.base.Wrapper

tracker
~~~~~~~
.. autosummary::
   :toctree: generated

   ixai.utils.tracker.SlidingWindowTracker
   ixai.utils.tracker.ExponentialSmoothingTracker
   ixai.utils.tracker.WelfordTracker
   ixai.utils.tracker.MultiValueTracker
   ixai.utils.tracker.base.Tracker

validators
~~~~~~~~~~
.. autosummary::
   :toctree: generated

   ixai.utils.validators.validate_loss_function
   ixai.utils.validators.validate_model_function
