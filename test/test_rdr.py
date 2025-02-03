from unittest import TestCase

import pandas as pd
from pandas.io import pickle
from typing_extensions import List
from ucimlrepo import fetch_ucirepo, dotdict

from pyrdr.datastructures import Case, Category, str_to_operator_fn, Condition
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
        scrdr = SingleClassRDR()
        cat = scrdr.classify(self.all_cases[0], Category(self.targets[0]))
        self.assertEqual(cat.name, self.targets[0])

    def test_fit_scrdr(self):
        scrdr = SingleClassRDR()
        scrdr.fit(self.all_cases, [Category(t) for t in self.targets], n_iter=20)
        scrdr.render_tree(use_dot_exporter=True, filename="scrdr")
        cat = scrdr.classify(self.all_cases[50])
        self.assertEqual(cat.name, self.targets[50])

    def test_classify_mcrdr(self):
        mcrdr = MultiClassRDR()
        cats = mcrdr.classify(self.all_cases[0], Category(self.targets[0]))
        self.assertEqual(cats[0].name, self.targets[0])

    def test_fit_mcrdr(self):
        self.expert_answer_idx = 0
        all_expert_conditions = self.mcrdr_test_expert()
        def expert(x, target, corner_case=None, diff_att=None):
            answer = all_expert_conditions[self.expert_answer_idx]
            self.expert_answer_idx += 1
            return answer
        mcrdr = MultiClassRDR()
        mcrdr.fit(self.all_cases, [Category(t) for t in self.targets],
                  ask_expert=expert)
        mcrdr.render_tree(use_dot_exporter=True, filename="mcrdr")
        cats = mcrdr.classify(self.all_cases[50])
        self.assertEqual(cats[0].name, self.targets[50])

    @staticmethod
    def mcrdr_test_expert():
        all_expert_answers = [{'milk': "milk == 1.0"},
                              {'aquatic': "aquatic == 1.0"},
                              {'feathers': "feathers == 1.0"},
                              {'backbone': "backbone == 0.0", 'breathes': "breathes == 0.0"},
                              {'backbone': "backbone == 0.0", 'fins': "fins == 0.0"},
                              {'milk': "milk == 1.0"},
                              {'feathers': "feathers == 1.0"},
                              {'eggs': "eggs == 1.0", 'breathes': "breathes == 1.0", 'backbone': "backbone == 0.0",
                               'milk': "milk == 0.0", 'fins': "fins == 0.0", 'aquatic': "aquatic == 0.0"},
                              {'breathes': "breathes == 1.0", 'fins': "fins == 0.0"},
                              {'milk': "milk == 1.0"},
                              {'backbone': "backbone == 0.0", 'aquatic': "aquatic == 0.0"},
                              {'feathers': "feathers == 1.0"},
                              {'tail': "tail == 1.0"},
                              {'milk': "milk == 1.0"},
                              {'feathers': "feathers == 1.0"},
                              {'backbone': "backbone == 0.0"},
                              {'aquatic': "aquatic == 1.0", 'breathes': "breathes == 0.0", 'fins': "fins == 1.0"},
                              {'fins': "fins == 0.0"},
                              {'legs': "legs == 0.0"},
                              {'tail': "tail == 0.0"},
                              {'backbone': "backbone == 0.0", 'breathes': "breathes == 1.0", 'fins': "fins == 0.0",
                               'feathers': "feathers == 0.0", 'milk': "milk == 0.0"}]
        all_expert_conditions = [{name: str_to_operator_fn(c) for name, c in a.items()} for a in all_expert_answers]
        all_expert_conditions = [{name: Condition(n, float(value), operator) for name, (n, value, operator) in a.items()}
                                 for a in all_expert_conditions]
        return all_expert_conditions
