import logging

__version__ = "0.6.46"

logger = logging.Logger("rdr")
logger.setLevel(logging.INFO)


# Trigger patch
try:
    from .datastructures.tracked_object import TrackedObjectMixin
    import ripple_down_rules_meta._apply_overrides
    print("OVERRIDEN")
except ImportError:
    print("IMPORTERROR")
