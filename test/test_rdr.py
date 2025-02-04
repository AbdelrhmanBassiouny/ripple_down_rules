import json
import os
from unittest import TestCase

import pandas as pd
from typing_extensions import List, Optional
from ucimlrepo import fetch_ucirepo, dotdict

from pyrdr.datastructures import Case, Category, str_to_operator_fn, Condition, MCRDRMode
from pyrdr.experts import Expert, Human
from pyrdr.helpers import create_cases_from_dataframe
from pyrdr.rdr import SingleClassRDR, MultiClassRDR


class TestRDR(TestCase):
    X: pd.DataFrame
    y: pd.DataFrame
    zoo: dotdict
    all_cases: List[Case]
    category_names: List[str]
    category_id_to_name: dict
    targets: List[str]

    @classmethod
    def setUpClass(cls):
        # fetch dataset
        cls.zoo = fetch_ucirepo(id=111)

        # data (as pandas dataframes)
        cls.X = cls.zoo.data.features
        cls.y = cls.zoo.data.targets
        # get ids as list of strings
        ids = cls.zoo.data.ids.values.flatten()
        cls.all_cases = create_cases_from_dataframe(cls.X, ids)
        # print category names
        cls.category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
        cls.category_id_to_name = {i + 1: name for i, name in enumerate(cls.category_names)}
        cls.targets = [cls.category_id_to_name[i] for i in cls.y.values.flatten()]

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_setup(self):
        self.assertEqual(self.X.shape, (101, 16))
        self.assertEqual(self.y.shape, (101, 1))
        all_rows = []
        for row in self.X.iterrows():
            all_rows.append(row[1])
        self.assertEqual(len(self.all_cases), 101)
        self.assertTrue(all([len(c.attributes) == 16 for c in self.all_cases]))
        self.assertTrue(all([isinstance(c.attributes, dict) for c in self.all_cases]))
        self.assertTrue(all([c.attribute_values == r.tolist()
                             for c, r in zip(self.all_cases, all_rows)]))

    def test_classify_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = "scrdr_expert_answers_classify"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        scrdr = SingleClassRDR()
        cat = scrdr.classify(self.all_cases[0], Category(self.targets[0]), expert=expert)
        self.assertEqual(cat.name, self.targets[0])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        draw_tree = False
        filename = "scrdr_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        scrdr = SingleClassRDR()
        scrdr.fit(self.all_cases, [Category(t) for t in self.targets], expert=expert,
                  draw_tree=draw_tree)
        scrdr.render_tree(use_dot_exporter=True, filename="scrdr")

        cat = scrdr.classify(self.all_cases[50])
        self.assertEqual(cat.name, self.targets[50])

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
        cats = mcrdr.classify(self.all_cases[0], Category(self.targets[0]), expert=expert)

        self.assertEqual(cats[0].name, self.targets[0])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_mcrdr_stop_only(self):
        draw_tree = False
        expert = MCRDRTester()
        mcrdr = MultiClassRDR()
        mcrdr.fit(self.all_cases, [Category(t) for t in self.targets],
                  expert=expert, draw_tree=draw_tree)
        mcrdr.render_tree(use_dot_exporter=True, filename="mcrdr")
        cats = mcrdr.classify(self.all_cases[50])
        self.assertEqual(cats[0].name, self.targets[50])

    def test_fit_mcrdr_stop_plus_rule(self):
        draw_tree = False
        expert = MCRDRTester(MCRDRMode.StopPlusRule)
        mcrdr = MultiClassRDR(mode=MCRDRMode.StopPlusRule)
        mcrdr.fit(self.all_cases, [Category(t) for t in self.targets],
                  expert=expert, draw_tree=draw_tree)
        mcrdr.render_tree(use_dot_exporter=True, filename="mcrdr")
        cats = mcrdr.classify(self.all_cases[50])
        self.assertEqual(cats[0].name, self.targets[50])

    def test_classify_mcrdr_with_extra_conclusions(self):
        use_loaded_answers = True
        save_answers = False
        file_name = "mcrdr_extra_expert_answers_classify"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(file_name)

        mcrdr = MultiClassRDR()
        cats = mcrdr.classify(self.all_cases[0], Category(self.targets[0]),
                              add_extra_conclusions=True, expert=expert)

        self.assertEqual(cats[0].name, self.targets[0])
        self.assertTrue(Category("lives only on land") in cats)

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
        mcrdr.fit(self.all_cases, [Category(t) for t in self.targets],
                  add_extra_conclusions=False, expert=expert, draw_tree=draw_tree)
        expert = Human(use_loaded_answers=use_loaded_answers)
        file_name = "mcrdr_extra_expert_answers_fit"
        if use_loaded_answers:
            expert.load_answers(file_name)
        mcrdr.fit(self.all_cases, [Category(t) for t in self.targets],
                  add_extra_conclusions=True, expert=expert, n_iter=10, draw_tree=draw_tree)
        cats = mcrdr.classify(self.all_cases[50])
        self.assertEqual(cats[0].name, self.targets[50])
        self.assertTrue(Category("lives only on land") in cats)
        mcrdr.render_tree(use_dot_exporter=True, filename="mcrdr_extra")
        if save_answers:
            expert.save_answers(file_name)


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
