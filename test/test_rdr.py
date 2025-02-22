import json
import os
from unittest import TestCase

from typing_extensions import List, Optional

from ripple_down_rules.datasets import load_zoo_dataset
from ripple_down_rules.datastructures import Case, str_to_operator_fn, Condition, MCRDRMode, Habitat, Attribute, \
    Attributes
from ripple_down_rules.experts import Expert, Human
from ripple_down_rules.rdr import SingleClassRDR, MultiClassRDR, GeneralRDR
from ripple_down_rules.utils import render_tree, import_class


class TestRDR(TestCase):
    all_cases: List[Case]
    targets: List[str]
    test_results_dir: str = "./test_results"
    expert_answers_dir: str = "./test_expert_answers"

    @classmethod
    def setUpClass(cls):
        # fetch dataset
        cls.all_cases, cls.targets = load_zoo_dataset()
        if not os.path.exists(cls.test_results_dir):
            os.makedirs(cls.test_results_dir)

    def test_setup(self):
        self.assertEqual(len(self.all_cases), 101)
        self.assertTrue(all([len(c.attributes) == 16 for c in self.all_cases]))
        self.assertTrue(all([isinstance(c.attributes, Attributes) for c in self.all_cases]))

    def test_classify_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = self.expert_answers_dir + "/scrdr_expert_answers_classify"
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

    def test_fit_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        draw_tree = False
        filename = self.expert_answers_dir + "/scrdr_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        scrdr = SingleClassRDR()
        scrdr.fit(self.all_cases, self.targets, expert=expert,
                  animate_tree=draw_tree)
        render_tree(scrdr.start_rule, use_dot_exporter=True,
                    filename=self.test_results_dir + f"/scrdr")

        cat = scrdr.classify(self.all_cases[50])
        self.assertEqual(cat, self.targets[50])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_classify_mcrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = self.expert_answers_dir + "/mcrdr_expert_answers_classify"
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
        use_loaded_answers = True
        draw_tree = False
        save_answers = False
        filename = self.expert_answers_dir + "/mcrdr_expert_answers_stop_only_fit"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)
        mcrdr = MultiClassRDR()
        mcrdr.fit(self.all_cases, self.targets,
                  expert=expert, animate_tree=draw_tree)
        render_tree(mcrdr.start_rule, use_dot_exporter=True,
                    filename=self.test_results_dir + f"/mcrdr_stop_only")
        cats = mcrdr.classify(self.all_cases[50])
        self.assertEqual(cats[0], self.targets[50])
        self.assertTrue(len(cats) == 1)
        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_mcrdr_stop_plus_rule(self):
        use_loaded_answers = True
        draw_tree = False
        save_answers = False
        filename = self.expert_answers_dir + "/mcrdr_stop_plus_rule_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)
        mcrdr = MultiClassRDR(mode=MCRDRMode.StopPlusRule)
        mcrdr.fit(self.all_cases, self.targets,
                  expert=expert, animate_tree=draw_tree)
        render_tree(mcrdr.start_rule, use_dot_exporter=True,
                    filename=self.test_results_dir + f"/mcrdr_stop_plus_rule")
        cats = mcrdr.classify(self.all_cases[50])
        self.assertEqual(cats[0], self.targets[50])
        self.assertTrue(len(cats) == 1)
        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_mcrdr_stop_plus_rule_combined(self):
        use_loaded_answers = True
        save_answers = False
        draw_tree = False
        filename = self.expert_answers_dir + "/mcrdr_stop_plus_rule_combined_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)
        mcrdr = MultiClassRDR(mode=MCRDRMode.StopPlusRuleCombined)
        mcrdr.fit(self.all_cases, self.targets,
                  expert=expert, animate_tree=draw_tree)
        render_tree(mcrdr.start_rule, use_dot_exporter=True,
                    filename=self.test_results_dir + f"/mcrdr_stop_plus_rule_combined")
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
        file_name = self.expert_answers_dir + "/mcrdr_extra_expert_answers_classify"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(file_name)

        mcrdr = MultiClassRDR()
        cats = mcrdr.fit_case(self.all_cases[0], self.targets[0],
                              add_extra_conclusions=True, expert=expert)
        render_tree(mcrdr.start_rule, use_dot_exporter=True,
                    filename=self.test_results_dir + f"/mcrdr_extra_classify")
        LivesOnlyOnLand = Attribute.get_subclass("LivesOnlyOnLand")
        self.assertEqual(cats, [self.targets[50], LivesOnlyOnLand(True)])

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
                  add_extra_conclusions=False, expert=expert, animate_tree=False)
        expert = Human(use_loaded_answers=use_loaded_answers)
        file_name = self.expert_answers_dir + "/mcrdr_extra_expert_answers_fit"
        if use_loaded_answers:
            expert.load_answers(file_name)
        mcrdr.fit(self.all_cases, self.targets,
                  add_extra_conclusions=True, expert=expert, n_iter=10, animate_tree=draw_tree)
        cats = mcrdr.classify(self.all_cases[50])
        LivesOnlyOnLand = Attribute.get_subclass("LivesOnlyOnLand")
        self.assertEqual(cats, [self.targets[50], LivesOnlyOnLand(True)])
        render_tree(mcrdr.start_rule, use_dot_exporter=True,
                    filename=self.test_results_dir + f"/mcrdr_extra")
        if save_answers:
            expert.save_answers(file_name)

    def test_classify_grdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = self.expert_answers_dir + "/grdr_expert_answers_classify"
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
        filename = self.expert_answers_dir + "/grdr_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        fit_scrdr = self.get_fit_scrdr(draw_tree=False)

        grdr = GeneralRDR({type(fit_scrdr.start_rule.conclusion): fit_scrdr})

        def get_habitat(x: Case, t: Attribute):
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
            render_tree(rule, use_dot_exporter=True,
                        filename=self.test_results_dir + f"/grdr_{type(rule.conclusion).__name__}")

        cats = grdr.classify(self.all_cases[50])
        self.assertEqual(cats, [self.targets[50], Habitat("land")])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_grdr_with_extra_conclusions(self):
        use_loaded_answers = True
        save_answers = False
        draw_tree = False
        filename = self.expert_answers_dir + "/grdr_expert_answers_fit_extra"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        fit_scrdr = self.get_fit_scrdr(draw_tree=False)

        grdr = GeneralRDR({type(fit_scrdr.start_rule.conclusion): fit_scrdr})

        grdr.fit(self.all_cases[17:], expert=expert,
                 animate_tree=draw_tree, n_iter=20, add_extra_conclusions=True)
        for rule in grdr.start_rules:
            render_tree(rule, use_dot_exporter=True,
                        filename=self.test_results_dir + f"/grdr_{type(rule.conclusion).__name__}")

        cats = grdr.classify(self.all_cases[50])
        self.assertEqual(cats, [self.targets[50], Habitat("land")])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def get_fit_scrdr(self, draw_tree=False) -> SingleClassRDR:
        filename = self.expert_answers_dir + "/scrdr_expert_answers_fit"
        expert = Human(use_loaded_answers=True)
        expert.load_answers(filename)

        scrdr = SingleClassRDR()
        scrdr.fit(self.all_cases, self.targets, expert=expert,
                  animate_tree=draw_tree)
        return scrdr


class MCRDRTester(Expert):

    def __init__(self, mode: MCRDRMode = MCRDRMode.StopOnly,
                 expert_answers_dir: str = "./test_expert_answers"):
        self.mode = mode
        self.expert_answers_dir = expert_answers_dir
        self.all_expert_answers = self.get_all_expert_answers(mode)
        self.current_answer_idx = 0

    def ask_for_conditions(self, x: Case, target: Attribute, last_evaluated_rule=None):
        answer = self.all_expert_answers[self.current_answer_idx]
        self.current_answer_idx += 1
        return answer

    def ask_for_extra_conclusions(self, x: Case, current_conclusions=None):
        pass

    def get_all_expert_answers(self, mode: MCRDRMode):
        if mode == MCRDRMode.StopPlusRule:
            json_file = self.expert_answers_dir + "/mcrdr_stop_plus_rule_answers_fit.json"
            with open(json_file, "r") as f:
                all_expert_answers = json.load(f)
        elif mode == MCRDRMode.StopOnly:
            json_file = self.expert_answers_dir + "/mcrdr_stop_only_answers_fit.json"
            with open(json_file, "r") as f:
                all_expert_answers = json.load(f)
        all_expert_conditions = [{name: str_to_operator_fn(c) for name, c in a.items()} for a in all_expert_answers]
        all_expert_conditions = [
            {name: Condition(n, float(value), operator) for name, (n, value, operator) in a.items()}
            for a in all_expert_conditions]
        return all_expert_conditions

    def ask_if_conclusion_is_correct(self, x: Case, conclusion: Attribute,
                                     target: Optional[Attribute] = None,
                                     current_conclusions: Optional[List[Attribute]] = None) -> bool:
        pass
