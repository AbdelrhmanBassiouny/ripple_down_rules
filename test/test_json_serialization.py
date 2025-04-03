import json
from unittest import TestCase

from typing_extensions import List

from ripple_down_rules.datasets import load_zoo_dataset
from ripple_down_rules.datastructures import CaseQuery, Case
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR, MultiClassRDR
from ripple_down_rules.utils import make_set
from test_helpers.helpers import get_fit_mcrdr, get_fit_scrdr


class TestJSONSerialization(TestCase):
    all_cases: List[Case]
    targets: List[str]
    cache_dir: str = "./test_results"
    expert_answers_dir: str = "./test_expert_answers"

    @classmethod
    def setUpClass(cls):
        cls.all_cases, cls.targets = load_zoo_dataset(cls.cache_dir + "/zoo_dataset.pkl")

    def test_scrdr_json_serialization(self):
        scrdr = get_fit_scrdr(self.all_cases, self.targets)
        filename = f"{self.cache_dir}/scrdr.json"
        scrdr.save(filename)
        scrdr = SingleClassRDR.load(filename)
        for case, target in zip(self.all_cases, self.targets):
            cat = scrdr.classify(case)
            self.assertEqual(cat, target)

    def test_mcrdr_json_serialization(self):
        mcrdr = get_fit_mcrdr(self.all_cases, self.targets)
        filename = f"{self.cache_dir}/mcrdr.json"
        mcrdr.save(filename)
        mcrdr = MultiClassRDR.load(filename)
        for case, target in zip(self.all_cases, self.targets):
            cat = mcrdr.classify(case)
            self.assertEqual(make_set(cat), make_set(target))
