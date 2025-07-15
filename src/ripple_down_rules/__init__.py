import logging

__version__ = "0.6.46"

logger = logging.Logger("rdr")
logger.setLevel(logging.INFO)


# Trigger patch
try:
    from .datastructures.tracked_object import TrackedObjectMixin
    from .rdr_decorators import RDRDecorator
    import ripple_down_rules_meta._apply_overrides
except ImportError:
    pass
