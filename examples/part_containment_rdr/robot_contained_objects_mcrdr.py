from ripple_down_rules.datastructures.case import Case, create_case
from typing_extensions import Set, Union
from ripple_down_rules.utils import make_set
from .robot_contained_objects_mcrdr_defs import *
from ripple_down_rules.rdr import MultiClassRDR


attribute_name = 'contained_objects'
conclusion_type = (PhysicalObject, set, list,)
type_ = MultiClassRDR


def classify(case: Robot) -> Set[PhysicalObject]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_292338421353147015053324365450552788888(case):
        conclusions.update(make_set(conclusion_292338421353147015053324365450552788888(case)))
    return conclusions
