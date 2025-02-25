from enum import Enum

import sqlalchemy
from sqlalchemy.orm import MappedAsDataclass, Mapped, mapped_column
from typing_extensions import Tuple, List
from ucimlrepo import fetch_ucirepo

from ripple_down_rules.datastructures import Case, Attribute, Species


def load_zoo_dataset() -> Tuple[List[Case], List[Attribute]]:
    """
    Load the zoo dataset.

    :return: all cases and targets.
    """
    # fetch dataset
    zoo = fetch_ucirepo(id=111)

    # data (as pandas dataframes)
    X = zoo.data.features
    y = zoo.data.targets
    # get ids as list of strings
    ids = zoo.data.ids.values.flatten()
    all_cases = Case.create_cases_from_dataframe(X, ids)

    category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
    category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
    targets = [Species(Species.Values.from_str(category_id_to_name[i])) for i in y.values.flatten()]
    return all_cases, targets



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
