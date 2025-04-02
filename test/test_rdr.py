import os
from unittest import TestCase, skip

from typing_extensions import List

from ripple_down_rules.datasets import Habitat, Species
from ripple_down_rules.datasets import load_zoo_dataset
from ripple_down_rules.datastructures import Case, MCRDRMode, \
    Row, Column, Category, CaseQuery
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR, MultiClassRDR, GeneralRDR
from ripple_down_rules.utils import render_tree, get_all_subclasses


class TestRDR(TestCase):
    all_cases: List[Case]
    targets: List[str]
    test_results_dir: str = "./test_results"
    expert_answers_dir: str = "./test_expert_answers"
    generated_rdrs_dir: str = "./test_generated_rdrs"
    cache_file: str = f"{test_results_dir}/zoo_dataset.pkl"

    @classmethod
    def setUpClass(cls):
        # fetch dataset
        cls.all_cases, cls.targets = load_zoo_dataset(cache_file=cls.cache_file)
        for test_dir in [cls.test_results_dir, cls.expert_answers_dir, cls.generated_rdrs_dir]:
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

    def tearDown(self):
        Row._registry = {}
        Column._registry = {}

    def test_classify_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = self.expert_answers_dir + "/scrdr_expert_answers_classify"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        scrdr = SingleClassRDR()
        cat = scrdr.fit_case(CaseQuery(self.all_cases[0], target=self.targets[0]), expert=expert)
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
        case_queries = [CaseQuery(case, target=target) for case, target in zip(self.all_cases, self.targets)]
        scrdr.fit(case_queries, expert=expert,
                  animate_tree=draw_tree)
        scrdr.write_to_python_file(self.generated_rdrs_dir + "/scrdr.py")
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
        cats = mcrdr.fit_case(CaseQuery(self.all_cases[0], target=self.targets[0]), expert=expert)

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
        case_queries = [CaseQuery(case, target=target) for case, target in zip(self.all_cases, self.targets)]
        mcrdr.fit(case_queries, expert=expert, animate_tree=draw_tree, n_iter=1)
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
        append = False
        filename = self.expert_answers_dir + "/mcrdr_stop_plus_rule_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)
        mcrdr = MultiClassRDR(mode=MCRDRMode.StopPlusRule)
        case_queries = [CaseQuery(case, target=target) for case, target in zip(self.all_cases, self.targets)]
        try:
            mcrdr.fit(case_queries, expert=expert, animate_tree=draw_tree)
        # catch pop from empty list error
        except IndexError as e:
            if append:
                expert.use_loaded_answers = False
                mcrdr.fit(case_queries, expert=expert, animate_tree=draw_tree)
            else:
                raise e
        render_tree(mcrdr.start_rule, use_dot_exporter=True,
                    filename=self.test_results_dir + f"/mcrdr_stop_plus_rule")
        cats = mcrdr.classify(self.all_cases[50])
        self.assertEqual(cats[0], self.targets[50])
        self.assertTrue(len(cats) == 1)
        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file, append=append)

    def test_fit_mcrdr_stop_plus_rule_combined(self):
        use_loaded_answers = True
        save_answers = False
        draw_tree = False
        append = False
        filename = self.expert_answers_dir + "/mcrdr_stop_plus_rule_combined_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)
        mcrdr = MultiClassRDR(mode=MCRDRMode.StopPlusRuleCombined)
        case_queries = [CaseQuery(case, target=target) for case, target in zip(self.all_cases, self.targets)]
        try:
            mcrdr.fit(case_queries, expert=expert, animate_tree=draw_tree)
        # catch pop from empty list error
        except IndexError as e:
            if append:
                expert.use_loaded_answers = False
                mcrdr.fit(case_queries, expert=expert, animate_tree=draw_tree)
            else:
                raise e
        render_tree(mcrdr.start_rule, use_dot_exporter=True,
                    filename=self.test_results_dir + f"/mcrdr_stop_plus_rule_combined")
        cats = mcrdr.classify(self.all_cases[50])
        self.assertEqual(cats[0], self.targets[50])
        self.assertTrue(len(cats) == 1)
        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file, append=append)

    @skip("Extra conclusions loaded answers are not working with new prompt interface")
    def test_classify_mcrdr_with_extra_conclusions(self):
        use_loaded_answers = True
        save_answers = False
        file_name = self.expert_answers_dir + "/mcrdr_extra_expert_answers_classify"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(file_name)

        mcrdr = MultiClassRDR()
        cats = mcrdr.fit_case(CaseQuery(self.all_cases[0], target=self.targets[0]),
                              add_extra_conclusions=True, expert=expert)
        render_tree(mcrdr.start_rule, use_dot_exporter=True,
                    filename=self.test_results_dir + f"/mcrdr_extra_classify")
        LivesOnlyOnLand = get_all_subclasses(Column)["LivesOnlyOnLand".lower()]
        self.assertEqual(cats, [self.targets[50], LivesOnlyOnLand(True)])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, file_name)
            expert.save_answers(file)

    @skip("Extra conclusions loaded answers are not working with new prompt interface")
    def test_fit_mcrdr_with_extra_conclusions(self):
        draw_tree = False
        use_loaded_answers = True
        save_answers = False
        expert = Human(use_loaded_answers=use_loaded_answers)
        mcrdr = MultiClassRDR()
        case_queries = [CaseQuery(case, target=target) for case, target in zip(self.all_cases, self.targets)]
        mcrdr.fit(case_queries, add_extra_conclusions=False, expert=expert, animate_tree=False)
        expert = Human(use_loaded_answers=use_loaded_answers)
        file_name = self.expert_answers_dir + "/mcrdr_extra_expert_answers_fit"
        if use_loaded_answers:
            expert.load_answers(file_name)
        mcrdr.fit(case_queries, add_extra_conclusions=True, expert=expert, n_iter=10, animate_tree=draw_tree)
        cats = mcrdr.classify(self.all_cases[50])
        LivesOnlyOnLand = get_all_subclasses(Column)["LivesOnlyOnLand".lower()]
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

        targets = [self.targets[0], Habitat.land]
        case_queries = [CaseQuery(self.all_cases[0], target=target) for target in targets]
        cats = grdr.fit_case(case_queries, expert=expert)
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

        def get_habitat(x: Row, t: Category):
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
            return all_habs + [t]

        n = 20
        habitat_targets = [get_habitat(x, t) for x, t in zip(self.all_cases[:n], self.targets[:n])]
        case_queries = [CaseQuery(case, target=target)
                        for case, targets in zip(self.all_cases[:n], habitat_targets)
                        for target in targets]
        grdr.fit(case_queries, expert=expert,
                 animate_tree=draw_tree, n_iter=n)
        for rule in grdr.start_rules:
            render_tree(rule, use_dot_exporter=True,
                        filename=self.test_results_dir + f"/grdr_{type(rule.conclusion).__name__}")

        cats = grdr.classify(self.all_cases[50])
        self.assertEqual(cats, [self.targets[50], Habitat.land])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    @skip("Extra conclusions loaded answers are not working with new prompt interface")
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
        self.assertEqual(cats, [self.targets[50], Habitat.land])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def get_fit_scrdr(self, draw_tree=False) -> SingleClassRDR:
        filename = self.expert_answers_dir + "/scrdr_expert_answers_fit"
        expert = Human(use_loaded_answers=True)
        expert.load_answers(filename)

        scrdr = SingleClassRDR()
        case_queries = [CaseQuery(case, target=target) for case, target in zip(self.all_cases, self.targets)]
        scrdr.fit(case_queries, expert=expert,
                  animate_tree=draw_tree)
        return scrdr
