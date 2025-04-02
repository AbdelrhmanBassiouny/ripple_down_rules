import json
from unittest import TestCase

from typing_extensions import List

from ripple_down_rules.datasets import load_zoo_dataset
from ripple_down_rules.datastructures import CaseQuery, Case
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR


class TestJSONSerialization(TestCase):
    all_cases: List[Case]
    targets: List[str]
    cache_dir: str = "./test_results"
    expert_answers_dir: str = "./test_expert_answers"

    @classmethod
    def setUpClass(cls):
        cls.all_cases, cls.targets = load_zoo_dataset(cls.cache_dir + "/zoo_dataset.pkl")

    def test_scrdr_json_serialization(self):
        scrdr = self.get_fit_scrdr()
        filename = f"{self.cache_dir}/scrdr.json"
        scrdr.save(filename)
        scrdr = SingleClassRDR.load(filename)
        for case, target in zip(self.all_cases, self.targets):
            cat = scrdr.classify(case)
            self.assertEqual(cat, target)

    def get_fit_scrdr(self, draw_tree=False) -> SingleClassRDR:
        filename = self.expert_answers_dir + "/scrdr_expert_answers_fit"
        expert = Human(use_loaded_answers=True)
        expert.load_answers(filename)

        scrdr = SingleClassRDR()
        case_queries = [CaseQuery(case, target=target) for case, target in zip(self.all_cases, self.targets)]
        scrdr.fit(case_queries, expert=expert,
                  animate_tree=draw_tree)
        for case, target in zip(self.all_cases, self.targets):
            cat = scrdr.classify(case)
            self.assertEqual(cat, target)
        return scrdr