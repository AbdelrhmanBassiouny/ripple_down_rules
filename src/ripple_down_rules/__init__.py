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

from .datastructures.dataclasses import CaseQuery
from .rdr_decorators import RDRDecorator
from .rdr import MultiClassRDR, SingleClassRDR, GeneralRDR