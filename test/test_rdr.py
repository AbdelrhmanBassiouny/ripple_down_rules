from unittest import TestCase

import pandas as pd
from typing_extensions import List
from ucimlrepo import fetch_ucirepo, dotdict

from episode_segmenter.helpers import create_cases_from_dataframe
from episode_segmenter.ripple_down_rules import Case, Attribute, SingleClassRDR, Category


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
        cls.category_id_to_name = {i+1: name for i, name in enumerate(cls.category_names)}
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
        scrdr.fit(self.all_cases, [Category(t) for t in self.targets])
        cat = scrdr.classify(self.all_cases[50])
        self.assertEqual(cat.name, self.targets[50])
