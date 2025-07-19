from ripple_down_rules.datastructures.case import Case, create_case
from typing_extensions import Optional, Set
from ripple_down_rules.utils import make_set
from ripple_down_rules.helpers import get_an_updated_case_copy
from .robot_get_contained_objects_output__mcrdr_defs import *


attribute_name = 'output_'
conclusion_type = (set, list, PhysicalObject,)
mutually_exclusive = False
name = 'output_'
case_type = Dict
case_name = 'Robot_get_contained_objects'


def classify(case: Dict, **kwargs) -> Set[PhysicalObject]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_167615852950279355863004646114673699744(case):
        conclusions.update(make_set(conclusion_167615852950279355863004646114673699744(case)))
    return conclusions
