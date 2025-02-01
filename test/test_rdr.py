from unittest import TestCase

import pandas as pd
from ucimlrepo import fetch_ucirepo, dotdict

from episode_segmenter.helpers import create_cases_from_dataframe
from episode_segmenter.ripple_down_rules import Case, Attribute


class TestRDR(TestCase):
    X: pd.DataFrame
    y: pd.DataFrame
    zoo: dotdict

    @classmethod
    def setUpClass(cls):
        # fetch dataset
        cls.zoo = fetch_ucirepo(id=111)

        # data (as pandas dataframes)
        cls.X = cls.zoo.data.features
        cls.y = cls.zoo.data.targets

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
        all_cases = create_cases_from_dataframe(self.X)
        self.assertEqual(len(all_cases), 101)
        self.assertTrue(all([len(c.attributes) == 16 for c in all_cases]))
        self.assertTrue(all([isinstance(c.attributes, dict) for c in all_cases]))
        self.assertTrue(all([c.attribute_values == r.tolist()
                             for c, r in zip(all_cases, all_rows)]))
