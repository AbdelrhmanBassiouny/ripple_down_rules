__version__ = "0.6.52"

import logging
import sys

logger = logging.Logger("rdr")
logger.setLevel(logging.INFO)

try:
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
except ImportError:
    pass

# Trigger patch
try:
    from .datastructures.tracked_object import TrackedObjectMixin
    import ripple_down_rules_meta._apply_overrides
    print("OVERRIDEN")
except ImportError:
    print("IMPORTERROR")

from .datastructures.dataclasses import CaseQuery
from .rdr_decorators import RDRDecorator
from .rdr import MultiClassRDR, SingleClassRDR, GeneralRDR