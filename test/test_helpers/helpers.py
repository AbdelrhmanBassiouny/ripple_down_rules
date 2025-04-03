from typing_extensions import List, Any

from ripple_down_rules.datastructures import CaseQuery
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import MultiClassRDR, SingleClassRDR
from ripple_down_rules.utils import make_set


def get_fit_scrdr(cases: List[Any], targets: List[Any], expert_answers_dir: str = "./test_expert_answers",
                  expert_answers_file: str = "/scrdr_expert_answers_fit",
                  draw_tree: bool = False) -> SingleClassRDR:
    filename = expert_answers_dir + expert_answers_file
    expert = Human(use_loaded_answers=True)
    expert.load_answers(filename)

    scrdr = SingleClassRDR()
    case_queries = [CaseQuery(case, target=target) for case, target in zip(cases, targets)]
    scrdr.fit(case_queries, expert=expert,
              animate_tree=draw_tree)
    for case, target in zip(cases, targets):
        cat = scrdr.classify(case)
        assert cat == target
    return scrdr


def get_fit_mcrdr(cases: List[Any], targets: List[Any], expert_answers_dir: str = "./test_expert_answers",
                  expert_answers_file: str = "/mcrdr_expert_answers_stop_only_fit",
                  draw_tree: bool = False):
    filename = expert_answers_dir + expert_answers_file
    expert = Human(use_loaded_answers=True)
    expert.load_answers(filename)
    mcrdr = MultiClassRDR()
    case_queries = [CaseQuery(case, target=target) for case, target in zip(cases, targets)]
    mcrdr.fit(case_queries, expert=expert, animate_tree=draw_tree)
    for case, target in zip(cases, targets):
        cat = mcrdr.classify(case)
        assert make_set(cat) == make_set(target)
    return mcrdr

