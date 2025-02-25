import os
from enum import Enum
from unittest import TestCase

import sqlalchemy.orm
from sqlalchemy import select
from sqlalchemy.orm import MappedAsDataclass, Mapped, mapped_column
from typing_extensions import List
from ucimlrepo import fetch_ucirepo

from ripple_down_rules.alchemy_rules import Target, AlchemyRule
from ripple_down_rules.datasets import load_zoo_dataset, Base, Animal, Species, get_dataset
from ripple_down_rules.datastructures import Case, Attributes, ObjectPropertyTarget, RDRMode
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR
from ripple_down_rules.utils import prompt_for_relational_conditions, prompt_for_alchemy_conditions
from test_rdr import TestRDR

class TestAlchemyRDR:
    session: sqlalchemy.orm.Session

    @classmethod
    def setUpClass(cls):
        zoo = get_dataset(111, TestRDR.cache_file)

        # data (as pandas dataframes)
        X = zoo['features']
        y = zoo['targets']
        # get ids as list of strings

        category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
        category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
        X.loc[:, "species"] = [Species(category_id_to_name[i]) for i in y.values.flatten()]

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = sqlalchemy.orm.Session(engine)
        session.bulk_insert_mappings(Animal, X.to_dict(orient="records"))
        session.commit()
        cls.session = session

    def test_alchemy_rules(self):
        rule1 = AlchemyRule(Animal.hair == True,
                            Target(Animal.species, Species.mammal))
        rule2 = AlchemyRule(Animal.feathers == True,
                            Target(Animal.species, Species.bird), parent=rule1)
        print(rule2.statement_of_path())
        query = rule1.statement.where(rule1.target.column != rule1.target.value)
        result = self.session.execute(query).first()
        print(result)

    def test_setup(self):
        r = self.session.scalars(select(Animal)).all()
        assert len(r) == 101

        user_input, conditions = prompt_for_alchemy_conditions(Animal, ObjectPropertyTarget(Animal, Animal.species, Species.mammal))
        print(conditions)
        print(type(conditions))

    def test_classify_scrdr(self):

        expert = Human(use_loaded_answers=False, mode=RDRMode.Relational)

        query = select(Animal)
        result = self.session.scalars(query).all()


        scrdr = SingleClassRDR(mode=RDRMode.Relational)
        scrdr.table = Animal
        scrdr.target_column = Animal.species

        cat = scrdr.fit_case(result[0], target=result[0].species, expert=expert)
        assert cat == result[0].species

tests = TestAlchemyRDR()
tests.setUpClass()
# tests.test_alchemy_rules()
tests.test_classify_scrdr()
