from typing_extensions import Any, Dict
from ripple_down_rules.helpers import general_rdr_classify
from ripple_down_rules.datastructures.case import Case, create_case
from relational_model import Robot
from . import robot_contained_objects_mcrdr as contained_objects_classifier

name = 'contained_objects'
case_type = Robot
case_name = 'Robot'
classifiers_dict = dict()
classifiers_dict['contained_objects'] = contained_objects_classifier


def classify(case: Robot, **kwargs) -> Dict[str, Any]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    return general_rdr_classify(classifiers_dict, case, **kwargs)
