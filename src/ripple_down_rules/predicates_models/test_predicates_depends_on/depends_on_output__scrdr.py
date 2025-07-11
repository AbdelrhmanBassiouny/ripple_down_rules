from ...datastructures.case import create_case
from types import NoneType
from typing_extensions import Optional
from .depends_on_output__scrdr_defs import *


attribute_name = 'output_'
conclusion_type = (bool,)
mutually_exclusive = True


def classify(case: Dict, **kwargs) -> Optional[bool]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)

    if conditions_15107258415760040561965086064322061396(case):
        return conclusion_15107258415760040561965086064322061396(case)
    else:
        return None
