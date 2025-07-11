from typing_extensions import Any, Dict
from ...helpers import general_rdr_classify
from ...datastructures.case import Case, create_case
from . import depends_on_output__scrdr as output__classifier

classifiers_dict = dict()
classifiers_dict['output_'] = output__classifier


def classify(case: Dict, **kwargs) -> Dict[str, Any]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    return general_rdr_classify(classifiers_dict, case, **kwargs)
