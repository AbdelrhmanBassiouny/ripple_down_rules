from __future__ import annotations

import os
import pickle

import sqlalchemy
from sqlalchemy import ForeignKey
from sqlalchemy.orm import MappedAsDataclass, Mapped, mapped_column, relationship
from typing_extensions import Tuple, List, Set, Optional
from ucimlrepo import fetch_ucirepo

from .datastructures.case import Case, create_cases_from_dataframe
from .datastructures.enums import Category


def load_cached_dataset(cache_file):
    """Loads the dataset from cache if it exists."""
    dataset = {}
    for key in ["features", "targets", "ids"]:
        part_file = cache_file.replace(".pkl", f"_{key}.pkl")
        if not os.path.exists(part_file):
            return None
        with open(part_file, "rb") as f:
            dataset[key] = pickle.load(f)
    return dataset


def save_dataset_to_cache(dataset, cache_file):
    """Saves only essential parts of the dataset to cache."""
    dataset_to_cache = {
        "features": dataset.data.features,
        "targets": dataset.data.targets,
        "ids": dataset.data.ids,
    }

    for key, value in dataset_to_cache.items():
        with open(cache_file.replace(".pkl", f"_{key}.pkl"), "wb") as f:
            pickle.dump(dataset_to_cache[key], f)
    print("Dataset cached successfully.")


def get_dataset(dataset_id, cache_file: Optional[str] = None):
    """Fetches dataset from cache or downloads it if not available."""
    dataset = load_cached_dataset(cache_file) if cache_file else None
    if dataset is None:
        print("Downloading dataset...")
        dataset = fetch_ucirepo(id=dataset_id)

        # Check if dataset is valid before caching
        if dataset is None or not hasattr(dataset, "data"):
            print("Error: Failed to fetch dataset.")
            return None

        if cache_file:
            save_dataset_to_cache(dataset, cache_file)

        dataset = {
            "features": dataset.data.features,
            "targets": dataset.data.targets,
            "ids": dataset.data.ids,
        }

    return dataset


def load_zoo_dataset(cache_file: Optional[str] = None) -> Tuple[List[Case], List[Species]]:
    """
    Load the zoo dataset.

    :param cache_file: the cache file to store the dataset or load it from.
    :return: all cases and targets.
    """
    # fetch dataset
    zoo = get_dataset(111, cache_file)

    # data (as pandas dataframes)
    X = zoo['features']
    y = zoo['targets']
    # get ids as list of strings
    ids = zoo['ids'].values.flatten()
    all_cases = create_cases_from_dataframe(X, name="Animal")

    category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
    category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
    # targets = [getattr(SpeciesCol, category_id_to_name[i]) for i in y.values.flatten()]
    targets = [Species.from_str(category_id_to_name[i]) for i in y.values.flatten()]
    return all_cases, targets


class Species(Category):
    mammal = "mammal"
    bird = "bird"
    reptile = "reptile"
    fish = "fish"
    amphibian = "amphibian"
    insect = "insect"
    molusc = "molusc"


class Habitat(Category):
    """
    A habitat category is a category that represents the habitat of an animal.
    """
    land = "land"
    water = "water"
    air = "air"


# SpeciesCol = Column.create_from_enum(Species, mutually_exclusive=True)
# HabitatCol = Column.create_from_enum(Habitat, mutually_exclusive=False)


class Base(sqlalchemy.orm.DeclarativeBase):
    pass


class HabitatTable(MappedAsDataclass, Base):
    __tablename__ = "Habitat"

    id: Mapped[int] = mapped_column(init=False, primary_key=True, autoincrement=True)
    habitat: Mapped[Habitat]
    animal_id = mapped_column(ForeignKey("Animal.id"), init=False)

    def __hash__(self):
        return hash(self.habitat)

    def __str__(self):
        return self.habitat.value

    def __repr__(self):
        return self.__str__()


class Animal(MappedAsDataclass, Base):
    __tablename__ = "Animal"

    id: Mapped[int] = mapped_column(init=False, primary_key=True, autoincrement=True)
    name: Mapped[str]
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
    species: Mapped[Species] = mapped_column(nullable=True)

    habitats: Mapped[Set[HabitatTable]] = relationship(default_factory=set)
