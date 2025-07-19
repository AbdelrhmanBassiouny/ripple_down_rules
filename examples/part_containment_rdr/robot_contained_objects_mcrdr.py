from ripple_down_rules.utils import make_set
from ripple_down_rules.helpers import get_an_updated_case_copy
from typing_extensions import Optional, Set
from ripple_down_rules.datastructures.case import Case, create_case
from .robot_contained_objects_mcrdr_defs import *


attribute_name = 'contained_objects'
conclusion_type = (set, PhysicalObject, list,)
mutually_exclusive = False
name = 'contained_objects'
case_type = Robot
case_name = 'Robot'


def classify(case: Robot, **kwargs) -> Set[PhysicalObject]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_298609776593271728826836208156881692889(case):
        conclusions.update(make_set(conclusion_298609776593271728826836208156881692889(case)))
    return conclusions
