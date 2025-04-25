from typing_extensions import List, Any, Tuple

from ripple_down_rules.datasets import Species, Habitat
from ripple_down_rules.datastructures.case import Case
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.datastructures.enums import Category
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import MultiClassRDR, SingleClassRDR, GeneralRDR
from ripple_down_rules.utils import make_set, is_iterable, flatten_list


def get_fit_scrdr(cases: List[Any], targets: List[Any], attribute_name: str = "species",
                  expert_answers_dir: str = "./test_expert_answers",
                  expert_answers_file: str = "/scrdr_expert_answers_fit",
                  draw_tree: bool = False) -> SingleClassRDR:
    filename = expert_answers_dir + expert_answers_file
    expert = Human(use_loaded_answers=True)
    expert.load_answers(filename)

    scrdr = SingleClassRDR()
    case_queries = [CaseQuery(case, attribute_name, target=target) for case, target in zip(cases, targets)]
    scrdr.fit(case_queries, expert=expert,
              animate_tree=draw_tree)
    for case, target in zip(cases, targets):
        cat = scrdr.classify(case)
        assert cat == target
    return scrdr


def get_fit_mcrdr(cases: List[Any], targets: List[Any], attribute_name: str = "species",
                  expert_answers_dir: str = "./test_expert_answers",
                  expert_answers_file: str = "/mcrdr_expert_answers_stop_only_fit",
                  draw_tree: bool = False):
    filename = expert_answers_dir + expert_answers_file
    expert = Human(use_loaded_answers=True)
    expert.load_answers(filename)
    mcrdr = MultiClassRDR()
    case_queries = [CaseQuery(case, attribute_name, target=target) for case, target in zip(cases, targets)]
    mcrdr.fit(case_queries, expert=expert, animate_tree=draw_tree)
    for case, target in zip(cases, targets):
        cat = mcrdr.classify(case)
        assert make_set(cat) == make_set(target)
    return mcrdr


def get_fit_grdr(cases: List[Any], targets: List[Any], expert_answers_dir: str = "./test_expert_answers",
                 expert_answers_file: str = "/grdr_expert_answers_fit", draw_tree: bool = False,
                 load_answers: bool = True) -> Tuple[GeneralRDR, List[dict]]:
    filename = expert_answers_dir + expert_answers_file
    expert = Human(use_loaded_answers=load_answers)
    if load_answers:
        expert.load_answers(filename)

    fit_scrdr = get_fit_scrdr(cases, targets, draw_tree=False)

    grdr = GeneralRDR()
    grdr.add_rdr(fit_scrdr)

    n = 20
    all_targets = [get_habitat(x, t) for x, t in zip(cases[:n], targets[:n])]
    case_queries = [CaseQuery(case, attribute_name, target=target)
                    for case, targets in zip(cases[:n], all_targets)
                    for attribute_name, target in targets.items()]
    grdr.fit(case_queries, expert=expert,
             animate_tree=draw_tree)
    for case, case_targets in zip(cases[:n], all_targets):
        cat = grdr.classify(case)
        for cat_name, cat_val in cat.items():
            if cat_name == "habitats":
                if "habitats" not in case_targets:
                    print(f"Case: {case}")
                assert make_set(cat_val) == make_set(case_targets[cat_name])
            else:
                assert cat_val == case_targets[cat_name]
    return grdr, all_targets


def get_habitat(x: Case, t: Category):
    habitat = set()
    if t == Species.mammal and x["aquatic"] == 0:
        habitat = {Habitat.land}
    elif t == Species.bird:
        habitat = {Habitat.land}
        if x["airborne"] == 1:
            habitat.update({Habitat.air})
        if x["aquatic"] == 1:
            habitat.update({Habitat.water})
    elif t == Species.fish:
        habitat = {Habitat.water}
    elif t == Species.molusc:
        habitat = {Habitat.land}
        if x["aquatic"] == 1:
            habitat.update({Habitat.water})
    if len(habitat) == 0:
        return {t.__class__.__name__.lower(): t}
    else:
        return {"habitats": habitat, t.__class__.__name__.lower(): t}
