from ripple_down_rules.datastructures.case import Case, create_case
from typing_extensions import Set, Union
from ripple_down_rules.utils import make_set
from .physical_object_select_objects_that_are_parts_of_robot_output__mcrdr_defs import *


attribute_name = 'output_'
conclusion_type = (set, PhysicalObject, list,)
mutually_exclusive = False


def classify(case: Dict, **kwargs) -> Set[PhysicalObject]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_164855806603893754507167918997373216146(case):
        conclusions.update(make_set(conclusion_164855806603893754507167918997373216146(case)))
    return conclusions
