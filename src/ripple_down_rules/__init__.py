__version__ = "0.6.47"

import logging
logger = logging.Logger("rdr")
logger.setLevel(logging.INFO)

from .datastructures.dataclasses import CaseQuery
from .rdr_decorators import RDRDecorator
from .rdr import MultiClassRDR, SingleClassRDR, GeneralRDR
