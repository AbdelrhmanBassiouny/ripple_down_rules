import os
from enum import Enum
from unittest import TestCase

import sqlalchemy.orm
from sqlalchemy import select
from sqlalchemy.orm import MappedAsDataclass, Mapped, mapped_column
from typing_extensions import List
from ucimlrepo import fetch_ucirepo

from ripple_down_rules.datasets import load_zoo_dataset
from ripple_down_rules.datastructures import Case, Attributes
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR

class Species(str, Enum):
    mammal = "mammal"
    bird = "bird"
    reptile = "reptile"
    fish = "fish"
    amphibian = "amphibian"
    insect = "insect"
    molusc = "molusc"


class Base(sqlalchemy.orm.DeclarativeBase):
    pass


class Animal(MappedAsDataclass, Base):
    __tablename__ = "Animal"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    hair: Mapped[bool]
    feathers: Mapped[bool]
    eggs: Mapped[bool]
    milk: Mapped[bool]
    airborne: Mapped[bool]
    aquatic: Mapped[bool]
    predator: Mapped[bool]
    toothed: Mapped[bool]
    backbone: Mapped[bool]
    breathes: Mapped[bool]
    venomous: Mapped[bool]
    fins: Mapped[bool]
    legs: Mapped[int]
    tail: Mapped[bool]
    domestic: Mapped[bool]
    catsize: Mapped[bool]
    species: Mapped[Species]


class TestRDR(TestCase):
    session: sqlalchemy.orm.Session

    @classmethod
    def setUpClass(cls):
        zoo = fetch_ucirepo(id=111)

        # data (as pandas dataframes)
        X = zoo.data.features
        y = zoo.data.targets
        # get ids as list of strings

        category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
        category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
        X["species"] = [Species(category_id_to_name[i]) for i in y.values.flatten()]

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = sqlalchemy.orm.Session(engine)
        session.bulk_insert_mappings(Animal, X.to_dict(orient="records"))
        session.commit()
        cls.session = session

    def test_setup(self):
        r = self.session.scalars(select(Animal)).all()
        self.assertEqual(len(r), 101)


    def test_classify_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = self.expert_answers_dir + "/scrdr_expert_answers_classify"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        scrdr = SingleClassRDR()
        cat = scrdr.fit_case(self.all_cases[0], self.targets[0], expert=expert)
        self.assertEqual(cat, self.targets[0])

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)
