import os
from unittest import TestCase

import sqlalchemy.orm
from sqlalchemy import select
from sqlalchemy.orm import MappedColumn as Column
from typing_extensions import List, Sequence

from ripple_down_rules.datasets import Base, Animal, Species, get_dataset, Habitat, HabitatTable
from ripple_down_rules.datastructures import CaseQuery
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR, MultiClassRDR, GeneralRDR
from ripple_down_rules.utils import render_tree, make_set


class TestAlchemyRDR(TestCase):
    session: sqlalchemy.orm.Session
    test_results_dir: str = "./test_results"
    expert_answers_dir: str = "./test_expert_answers"
    cache_file: str = f"{test_results_dir}/zoo_dataset.pkl"
    all_cases: Sequence[Animal]
    targets: List[Species]

    @classmethod
    def setUpClass(cls):
        zoo = get_dataset(111, cls.cache_file)

        # data (as pandas dataframes)
        X = zoo['features']
        y = zoo['targets']
        names = zoo['ids'].values.flatten()
        X.loc[:, "name"] = names

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = sqlalchemy.orm.Session(engine)
        session.bulk_insert_mappings(Animal, X.to_dict(orient="records"))
        session.commit()
        cls.session = session
        query = select(Animal)
        cls.all_cases = cls.session.scalars(query).all()
        category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
        category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
        cls.targets = [Species(category_id_to_name[i]) for i in y.values.flatten()]

    def test_fit_scrdr(self):
        use_loaded_answers = True
        draw_tree = False
        filename = self.expert_answers_dir + "/scrdr_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers, session=self.session)
        if use_loaded_answers:
            expert.load_answers(filename)

        query = select(Animal)
        result = self.session.scalars(query).all()
        scrdr = SingleClassRDR(session=self.session)
        case_queries = [CaseQuery(c, target=t) for c, t in zip(self.all_cases, self.targets)]
        scrdr.fit(case_queries, expert=expert,
                  animate_tree=draw_tree, session=self.session)

        cat = scrdr.classify(result[50])
        assert cat == self.targets[50]

    def test_fit_mcrdr_stop_only(self):
        use_loaded_answers = True
        draw_tree = False
        expert, filename = self.get_expert_and_file_name(use_loaded_answers,
                                                         "mcrdr_expert_answers_stop_only_fit")

        mcrdr = MultiClassRDR(session=self.session)
        case_queries = [CaseQuery(c, target=t) for c, t in zip(self.all_cases, self.targets)]
        mcrdr.fit(case_queries, expert=expert, animate_tree=draw_tree)

        cats = mcrdr.classify(self.all_cases[50])
        assert cats[0] == self.targets[50]
        assert len(cats) == 1

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

        def get_habitat(x: Animal, t: Column) -> List[Column]:
            all_habs = []
            if t == Species.mammal and x.aquatic == 0:
                all_habs.append(HabitatTable(Habitat.land))
            elif t == Species.bird:
                all_habs.append(HabitatTable(Habitat.land))
                if x.airborne == 1:
                    all_habs[-1] = make_set([all_habs[-1], HabitatTable(Habitat.air)])
                if x.aquatic == 1:
                    all_habs[-1] = make_set([all_habs[-1], HabitatTable(Habitat.water)])
            elif t == Species.fish:
                all_habs.append(HabitatTable(Habitat.water))
            elif t == Species.molusc:
                all_habs.append(HabitatTable(Habitat.land))
                if x.aquatic == 1:
                    all_habs[-1] = make_set([all_habs[-1], HabitatTable(Habitat.water)])
            atts = [x.habitats for _ in all_habs]
            atts.extend([x.species for _ in [t]])
            return all_habs + [t], atts

        n = 20
        habitat_targets = [get_habitat(x, t) for x, t in zip(self.all_cases[:n], self.targets[:n])]
        all_attributes = [h[1] for h in habitat_targets]
        habitat_targets = [h[0] for h in habitat_targets]
        case_queries = []
        for case, attributes, targets in zip(self.all_cases[:n], all_attributes, habitat_targets):
            for attr, target in zip(attributes, targets):
                case_queries.append(CaseQuery(case, attr, target=target))
        grdr.fit(case_queries, expert=expert, animate_tree=draw_tree)
        for rule in grdr.start_rules:
            render_tree(rule, use_dot_exporter=True,
                        filename=self.test_results_dir + f"/grdr_{type(rule.conclusion).__name__}")

        cats = grdr.classify(self.all_cases[50])
        assert cats == [self.targets[50], HabitatTable(Habitat.land)]

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def get_fit_scrdr(self, draw_tree=False) -> SingleClassRDR:
        filename = self.expert_answers_dir + "/scrdr_expert_answers_fit"
        expert = Human(use_loaded_answers=True)
        expert.load_answers(filename)

        scrdr = SingleClassRDR()
        case_queries = [CaseQuery(c, target=t) for c, t in zip(self.all_cases, self.targets)]
        scrdr.fit(case_queries, expert=expert, animate_tree=draw_tree)
        return scrdr

    def get_expert_and_file_name(self, use_loaded_answers: bool, filename: str):
        filename = self.expert_answers_dir + "/" + filename
        expert = Human(use_loaded_answers=use_loaded_answers, session=self.session)
        if use_loaded_answers:
            expert.load_answers(filename)
        return expert, filename


# tests = TestAlchemyRDR()
# tests.setUpClass()
# tests.test_fit_scrdr()
# tests.test_fit_mcrdr_stop_only()
# tests.test_fit_grdr()
