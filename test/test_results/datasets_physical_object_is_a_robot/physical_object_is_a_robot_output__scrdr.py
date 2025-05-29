from ripple_down_rules.datastructures.case import Case, create_case
from .physical_object_is_a_robot_output__scrdr_defs import *
from ripple_down_rules.rdr import SingleClassRDR


attribute_name = 'output_'
conclusion_type = (bool,)
mutually_exclusive = True


def classify(case: Dict) -> bool:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)

    if conditions_226969243620390858682731042391766665817(case):
        return conclusion_226969243620390858682731042391766665817(case)
