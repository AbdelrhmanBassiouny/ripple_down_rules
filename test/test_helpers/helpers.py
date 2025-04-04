from typing_extensions import List, Any

from ripple_down_rules.datasets import Species, Habitat
from ripple_down_rules.datastructures import CaseQuery, Case, Category
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import MultiClassRDR, SingleClassRDR, GeneralRDR
from ripple_down_rules.utils import make_set, is_iterable, flatten_list


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


def get_fit_grdr(cases: List[Any], targets: List[Any], expert_answers_dir: str = "./test_expert_answers",
                 expert_answers_file: str = "/grdr_expert_answers_fit", draw_tree: bool = False):
    filename = expert_answers_dir + expert_answers_file
    expert = Human(use_loaded_answers=True)
    expert.load_answers(filename)

    fit_scrdr = get_fit_scrdr(cases, targets, draw_tree=False)

    grdr = GeneralRDR({type(fit_scrdr.start_rule.conclusion): fit_scrdr})

    def get_habitat(x: Case, t: Category):
        all_habs = []
        if t == Species.mammal and x["aquatic"] == 0:
            all_habs.append({Habitat.land})
        elif t == Species.bird:
            all_habs.append({Habitat.land})
            if x["airborne"] == 1:
                all_habs[-1].update({Habitat.air})
            if x["aquatic"] == 1:
                all_habs[-1].update({Habitat.water})
        elif t == Species.fish:
            all_habs.append({Habitat.water})
        elif t == Species.molusc:
            all_habs.append({Habitat.land})
            if x["aquatic"] == 1:
                all_habs[-1].update({Habitat.water})
        return all_habs

    n = 20
    all_targets = [get_habitat(x, t) + [t] for x, t in zip(cases[:n], targets[:n])]
    case_queries = [CaseQuery(case, target=target)
                    for case, targets in zip(cases[:n], all_targets)
                    for target in targets]
    grdr.fit(case_queries, expert=expert,
             animate_tree=draw_tree, n_iter=n)
    for case, case_targets in zip(cases[:n], all_targets):
        cat = grdr.classify(case)
        cat = flatten_list(cat)
        case_targets = flatten_list(case_targets)
        assert make_set(cat) == make_set(case_targets)
    return grdr, all_targets
