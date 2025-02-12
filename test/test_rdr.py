import json
import os
from unittest import TestCase

from random_events.product_algebra import SimpleEvent
from random_events.variable import Symbolic
from typing_extensions import List, Optional
from ucimlrepo import fetch_ucirepo

from ripple_down_rules.datasets import load_zoo_dataset, infer_variables_from_dataframe
from ripple_down_rules.datastructures import Case, Category, str_to_operator_fn, Condition, MCRDRMode, Habitat, Equal
from ripple_down_rules.experts import Expert, Human
from ripple_down_rules.rdr import SingleClassRDR, MultiClassRDR, GeneralRDR
from ripple_down_rules.utils import render_tree


class TestRDR(TestCase):
    all_cases: List[Case]
    targets: List[str]

    @classmethod
    def setUpClass(cls):
        # fetch dataset
        cls.all_cases, cls.targets = load_zoo_dataset()

    def test_setup(self):
        self.assertEqual(len(self.all_cases), 101)
        self.assertTrue(all([len(c.attributes) == 16 for c in self.all_cases]))
        self.assertTrue(all([isinstance(c.attributes, dict) for c in self.all_cases]))

    def test_classify_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = "scrdr_expert_answers_classify"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        scrdr = SingleClassRDR()
        cat = scrdr.fit_case(self.all_cases[0], self.targets[0], expert=expert)
        self.assertEqual(cat, self.targets[0])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_small_scrdr_with_random_events(self):
        use_loaded_answers = True
        save_answers = False
        draw_tree = False
        n = 5
        filename = "small_scrdr_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        scrdr = SingleClassRDR()
        scrdr.fit(self.all_cases[:n], self.targets[:n], expert=expert,
                  animate_tree=draw_tree)
        render_tree(scrdr.start_rule, use_dot_exporter=True, filename="scrdr")

        cat = scrdr.classify(self.all_cases[2])
        self.assertEqual(cat, self.targets[2])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

        data = fetch_ucirepo(id=111)
        features = data.data.features
        print("=" * 80)
        for f in features.columns:
            if f != "legs":
                features.loc[:, (f, )] = features[f].apply(lambda x: str(bool(x)))
        targets = data.data.targets
        category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
        targets.loc[:, ("type", )] = targets["type"].apply(lambda x: category_names[x - 1])
        variables = infer_variables_from_dataframe(features) + infer_variables_from_dataframe(targets)
        name_to_variable = {v.name: v for v in variables}

        kb = SimpleEvent({v: v.domain for v in variables})
        kb = kb.as_composite_set()

        for node in (scrdr.start_rule,) + scrdr.start_rule.descendants:
            left = node.conditions
            left_event = SimpleEvent()
            for k,v in left.items():
                v: Condition
                if isinstance(v.operator, Equal):
                    variable: Symbolic = name_to_variable[v.name]
                    left_event[variable] = variable.make_value(str(bool(v.value)))

            # restate left => right as !left or right

            # create ~left
            left_event = left_event.__deepcopy__().as_composite_set()
            not_left = ~left_event
            not_left.fill_missing_variables(variables)

            # create right
            right_event = SimpleEvent()
            right = node.conclusion
            variable: Symbolic = name_to_variable["type"]
            right_event[variable] = variable.make_value(right.value)
            right_event = right_event.as_composite_set()
            right_event.fill_missing_variables(variables)

            rule = right_event.__deepcopy__() | not_left.__deepcopy__()

            kb = kb & rule

        model = ...
        # model.conditional(kb)

    def test_fit_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        draw_tree = False
        filename = "scrdr_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        scrdr = SingleClassRDR()
        scrdr.fit(self.all_cases, self.targets, expert=expert,
                  animate_tree=draw_tree)
        render_tree(scrdr.start_rule, use_dot_exporter=True, filename="scrdr")

        cat = scrdr.classify(self.all_cases[50])
        self.assertEqual(cat, self.targets[50])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_classify_mcrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = "mcrdr_expert_answers_classify"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        mcrdr = MultiClassRDR()
        cats = mcrdr.fit_case(self.all_cases[0], self.targets[0], expert=expert)

        self.assertEqual(cats[0], self.targets[0])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_mcrdr_stop_only(self):
        draw_tree = False
        expert = MCRDRTester()
        mcrdr = MultiClassRDR()
        mcrdr.fit(self.all_cases, self.targets,
                  expert=expert, animate_tree=draw_tree)
        render_tree(mcrdr.start_rule, use_dot_exporter=True, filename="mcrdr_stop_only")
        cats = mcrdr.classify(self.all_cases[50])
        self.assertEqual(cats[0], self.targets[50])
        self.assertTrue(len(cats) == 1)

    def test_fit_mcrdr_stop_plus_rule(self):
        draw_tree = False
        expert = MCRDRTester(MCRDRMode.StopPlusRule)
        mcrdr = MultiClassRDR(mode=MCRDRMode.StopPlusRule)
        mcrdr.fit(self.all_cases, self.targets,
                  expert=expert, animate_tree=draw_tree)
        render_tree(mcrdr.start_rule, use_dot_exporter=True, filename="mcrdr_stop_plus_rule")
        cats = mcrdr.classify(self.all_cases[50])
        self.assertEqual(cats[0], self.targets[50])
        self.assertTrue(len(cats) == 1)

    def test_fit_mcrdr_stop_plus_rule_combined(self):
        use_loaded_answers = True
        save_answers = False
        draw_tree = False
        filename = "mcrdr_stop_plus_rule_combined_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)
        mcrdr = MultiClassRDR(mode=MCRDRMode.StopPlusRuleCombined)
        mcrdr.fit(self.all_cases, self.targets,
                  expert=expert, animate_tree=draw_tree)
        render_tree(mcrdr.start_rule, use_dot_exporter=True, filename="mcrdr_stop_plus_rule_combined")
        cats = mcrdr.classify(self.all_cases[50])
        self.assertEqual(cats[0], self.targets[50])
        self.assertTrue(len(cats) == 1)
        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_classify_mcrdr_with_extra_conclusions(self):
        use_loaded_answers = True
        save_answers = False
        file_name = "mcrdr_extra_expert_answers_classify"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(file_name)

        mcrdr = MultiClassRDR()
        cats = mcrdr.fit_case(self.all_cases[0], self.targets[0],
                              add_extra_conclusions=True, expert=expert)
        render_tree(mcrdr.start_rule, use_dot_exporter=True, filename="mcrdr_extra_classify")

        self.assertEqual(cats, [self.targets[50], Category("LivesOnlyOnLand")])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, file_name)
            expert.save_answers(file)

    def test_fit_mcrdr_with_extra_conclusions(self):
        draw_tree = False
        use_loaded_answers = True
        save_answers = False
        expert = MCRDRTester()
        mcrdr = MultiClassRDR()
        mcrdr.fit(self.all_cases, self.targets,
                  add_extra_conclusions=False, expert=expert, animate_tree=draw_tree)
        expert = Human(use_loaded_answers=use_loaded_answers)
        file_name = "mcrdr_extra_expert_answers_fit"
        if use_loaded_answers:
            expert.load_answers(file_name)
        mcrdr.fit(self.all_cases, self.targets,
                  add_extra_conclusions=True, expert=expert, n_iter=10, animate_tree=draw_tree)
        cats = mcrdr.classify(self.all_cases[50])
        self.assertEqual(cats, [self.targets[50], Category("LivesOnlyOnLand")])
        render_tree(mcrdr.start_rule, use_dot_exporter=True, filename="mcrdr_extra")
        if save_answers:
            expert.save_answers(file_name)

    def test_classify_grdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = "grdr_expert_answers_classify"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        grdr = GeneralRDR()

        targets = [self.targets[0], Habitat("land")]
        cats = grdr.fit_case(self.all_cases[0], targets, expert=expert)
        self.assertEqual(cats, targets)

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_grdr(self):
        use_loaded_answers = True
        save_answers = False
        draw_tree = False
        filename = "grdr_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        fit_scrdr = self.get_fit_scrdr(draw_tree=False)

        grdr = GeneralRDR({type(fit_scrdr.start_rule.conclusion): fit_scrdr})

        def get_habitat(x: Case, t: Category):
            all_habs = []
            if t.value == "mammal" and x["aquatic"].value == 0:
                all_habs.append(Habitat("land"))
            elif t.value == "bird":
                all_habs.append(Habitat({"land"}))
                if x["airborne"].value == 1:
                    all_habs[-1].value.update({"air"})
                if x["aquatic"].value == 1:
                    all_habs[-1].value.update({"water"})
            elif t.value == "fish":
                all_habs.append(Habitat("water"))
            elif t.value == "molusc":
                all_habs.append(Habitat({"land"}))
                if x["aquatic"].value == 1:
                    all_habs[-1].value.update({"water"})
            return all_habs + [t]

        n = 20
        habitat_targets = [get_habitat(x, t) for x, t in zip(self.all_cases[:n], self.targets[:n])]
        grdr.fit(self.all_cases, habitat_targets, expert=expert,
                 animate_tree=draw_tree, n_iter=n)
        for rule in grdr.start_rules:
            render_tree(rule, use_dot_exporter=True, filename=f"grdr_{type(rule.conclusion).__name__}")

        cats = grdr.classify(self.all_cases[50])
        self.assertEqual(cats, [self.targets[50], Habitat("land")])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_grdr_with_extra_conclusions(self):
        use_loaded_answers = True
        save_answers = False
        draw_tree = True
        filename = "grdr_expert_answers_fit_extra"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        fit_scrdr = self.get_fit_scrdr(draw_tree=True)

        grdr = GeneralRDR({type(fit_scrdr.start_rule.conclusion): fit_scrdr})

        grdr.fit(self.all_cases[17:], expert=expert,
                 animate_tree=draw_tree, n_iter=20, add_extra_conclusions=True)
        for rule in grdr.start_rules:
            render_tree(rule, use_dot_exporter=True, filename=f"grdr_{type(rule.conclusion).__name__}")

        cats = grdr.classify(self.all_cases[50])
        self.assertEqual(cats, [self.targets[50], Habitat("land")])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def get_fit_scrdr(self, draw_tree=False) -> SingleClassRDR:
        filename = "scrdr_expert_answers_fit"
        expert = Human(use_loaded_answers=True)
        expert.load_answers(filename)

        scrdr = SingleClassRDR()
        scrdr.fit(self.all_cases, self.targets, expert=expert,
                  animate_tree=draw_tree, keep_open=False)
        return scrdr


class MCRDRTester(Expert):

    def __init__(self, mode: MCRDRMode = MCRDRMode.StopOnly):
        self.mode = mode
        self.all_expert_answers = self.get_all_expert_answers(mode)
        self.current_answer_idx = 0

    def ask_for_conditions(self, x: Case, target: Category, last_evaluated_rule=None):
        answer = self.all_expert_answers[self.current_answer_idx]
        self.current_answer_idx += 1
        return answer

    def ask_for_extra_conclusions(self, x: Case, current_conclusions=None):
        pass

    @staticmethod
    def get_all_expert_answers(mode: MCRDRMode):
        if mode == MCRDRMode.StopPlusRule:
            json_file = os.path.join(os.getcwd(), "mcrdr_stop_plus_rule_answers_fit.json")
            with open(json_file, "r") as f:
                all_expert_answers = json.load(f)
        elif mode == MCRDRMode.StopOnly:
            json_file = os.path.join(os.getcwd(), "mcrdr_stop_only_answers_fit.json")
            with open(json_file, "r") as f:
                all_expert_answers = json.load(f)
        all_expert_conditions = [{name: str_to_operator_fn(c) for name, c in a.items()} for a in all_expert_answers]
        all_expert_conditions = [
            {name: Condition(n, float(value), operator) for name, (n, value, operator) in a.items()}
            for a in all_expert_conditions]
        return all_expert_conditions

    def ask_if_conclusion_is_correct(self, x: Case, conclusion: Category,
                                     target: Optional[Category] = None,
                                     current_conclusions: Optional[List[Category]] = None) -> bool:
        pass
