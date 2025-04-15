import _ast
import importlib
import os
from _ast import AST
from enum import Enum
from unittest import TestCase, skip

from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.nx.helper import fully_factorized
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Symbolic
from typing_extensions import List

from ripple_down_rules.datasets import Habitat, Species, get_dataset
from ripple_down_rules.datasets import load_zoo_dataset
from ripple_down_rules.datastructures import Case, MCRDRMode, \
    Case, CaseAttribute, Category, CaseQuery
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR, MultiClassRDR, GeneralRDR
from ripple_down_rules.utils import render_tree, get_all_subclasses, make_set, flatten_list
from test_helpers.helpers import get_fit_scrdr, get_fit_mcrdr, get_fit_grdr


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

    def test_random_events(self):
        scrdr = get_fit_scrdr(self.all_cases, self.targets)

        # fetch dataset
        zoo = get_dataset(111, self.cache_file)

        # data (as pandas dataframes)
        features = zoo['features']
        targets = zoo['targets']

        print("=" * 80)
        for f in features.columns:
            features.loc[:, (f,)] = features[f].apply(lambda x: str(bool(x)))
        category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
        targets.loc[:, ("type",)] = targets["type"].apply(lambda x: category_names[x - 1])
        target_variable = Symbolic("type", Set.from_iterable(Species))
        variables = infer_variables_from_dataframe(features) + [target_variable]
        name_to_variable = {v.name: v for v in variables}

        kb = SimpleEvent({v: v.domain for v in variables}).as_composite_set()
        path_condition = kb.__deepcopy__()
        for node in (scrdr.start_rule,) + scrdr.start_rule.descendants:
            left = node.conditions
            left_event = SimpleEvent()
            for name, operator, value in left.visitor.compares:
                if isinstance(operator, _ast.Eq):
                    variable: Symbolic = name_to_variable[name.id]
                    left_event[variable] = variable.make_value(value.value)

            # restate left => right as !left or right

            # create ~left
            left_event = left_event.as_composite_set()
            not_left = ~left_event
            not_left.fill_missing_variables(variables)

            left_event = path_condition & left_event

            # create right
            right_event = SimpleEvent()
            right = node.conclusion
            variable: Symbolic = name_to_variable["type"]
            right_event[variable] = variable.make_value([right] if isinstance(right, (str, Enum)) else right)
            right_event = right_event.as_composite_set()
            right_event.fill_missing_variables(variables)

            rule = right_event | ~left_event
            kb = kb & rule
            path_condition = path_condition & not_left

        print(*kb.simple_sets, sep="\n")
        model = fully_factorized(variables, means={}, variances={})
        print(model)
        cooler_model, p = model.conditional(kb)
        print(cooler_model)
        print(p)

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
        render_tree(scrdr.start_rule, use_dot_exporter=True,
                    filename=self.test_results_dir + f"/scrdr")

        cat = scrdr.classify(self.all_cases[50])
        self.assertEqual(cat, self.targets[50])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_write_scrdr_to_python_file(self):
        scrdr = get_fit_scrdr(self.all_cases,self.targets)
        scrdr.write_to_python_file(self.generated_rdrs_dir)
        classify_species_scrdr = scrdr.get_rdr_classifier_from_python_file(self.generated_rdrs_dir)
        for case, target in zip(self.all_cases, self.targets):
            cat = classify_species_scrdr(case)
            self.assertEqual(cat, target)

    def test_write_mcrdr_to_python_file(self):
        mcrdr = get_fit_mcrdr(self.all_cases, self.targets)
        mcrdr.write_to_python_file(self.generated_rdrs_dir)
        classify_species_mcrdr = mcrdr.get_rdr_classifier_from_python_file(self.generated_rdrs_dir)
        for case, target in zip(self.all_cases, self.targets):
            cat = classify_species_mcrdr(case)
            self.assertEqual(make_set(cat), make_set(target))

    def test_write_grdr_to_python_file(self):
        grdr, all_targets = get_fit_grdr(self.all_cases, self.targets)
        grdr.write_to_python_file(self.generated_rdrs_dir)
        classify_animal_grdr = grdr.get_rdr_classifier_from_python_file(self.generated_rdrs_dir)
        for case, case_targets in zip(self.all_cases[:len(all_targets)], all_targets):
            cat = classify_animal_grdr(case)
            cat = flatten_list(cat)
            case_targets = flatten_list(case_targets)
            self.assertEqual(make_set(cat), make_set(case_targets))

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
        mcrdr.fit(case_queries, expert=expert, animate_tree=draw_tree)
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
        LivesOnlyOnLand = get_all_subclasses(CaseAttribute)["LivesOnlyOnLand".lower()]
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
        LivesOnlyOnLand = get_all_subclasses(CaseAttribute)["LivesOnlyOnLand".lower()]
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

        fit_scrdr = get_fit_scrdr(self.all_cases, self.targets, draw_tree=False)

        grdr = GeneralRDR({type(fit_scrdr.start_rule.conclusion): fit_scrdr})

        def get_habitat(x: Case, t: Category):
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

        fit_scrdr = get_fit_scrdr(self.all_cases, self.targets, draw_tree=False)

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
