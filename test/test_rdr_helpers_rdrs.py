from os.path import dirname, join

from typing_extensions import Union, Any, Dict

from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.rdr_decorators import RDRDecorator
from ripple_down_rules.utils import is_iterable, make_list

# app = QApplication(sys.argv)
save_dir = join(dirname(__file__), '..', 'src', 'ripple_down_rules')
# viewer = RDRCaseViewer(save_dir=save_dir)
viewer = None
rdr_decorator: RDRDecorator = RDRDecorator(save_dir, (bool,), True,
                                           fit=False,
                                           viewer=viewer)


@rdr_decorator.decorator
def should_i_ask_the_expert_for_a_target(conclusions: Union[Any, Dict[str, Any]],
                                         case_query: CaseQuery,
                                         update_existing: bool) -> bool:
    """
    Determine if the rdr should ask the expert for the target of a given case query.

    :param conclusions: The conclusions of the case.
    :param case_query: The query containing the case to classify.
    :param ask_always: Whether to ask the expert always.
    :param update_existing: Whether to update rules that gave the required type of conclusions.
    :return: True if the rdr should ask the expert, False otherwise.
    """
    if conclusions is None:
        return True
    elif is_iterable(conclusions) and len(conclusions) == 0:
        return True
    elif isinstance(conclusions, dict):
        if case_query.attribute_name not in conclusions:
            return True
        conclusions = conclusions[case_query.attribute_name]
    conclusion_types = map(type, make_list(conclusions))
    if not any(ct in case_query.core_attribute_type for ct in conclusion_types):
        return True
    elif update_existing:
        return True
    else:
        return False


def test_should_i_ask_the_expert_for_a_target(conclusions: Union[Any, Dict[str, Any]],
                                              case_query: CaseQuery,
                                              ask_always: bool,
                                              update_existing: bool):
    out = should_i_ask_the_expert_for_a_target(conclusions, case_query, ask_always, update_existing)
    # if not out:
    #     rdr_decorator.fit = True
    #     should_i_ask_the_expert_for_a_target(conclusions, case_query, ask_always, update_existing)
    # rdr_decorator.fit = False
