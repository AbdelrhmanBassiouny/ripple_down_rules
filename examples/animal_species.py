from __future__ import annotations
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.datastructures.case import create_cases_from_dataframe
from ripple_down_rules.rdr import GeneralRDR
from ucimlrepo import fetch_ucirepo
from enum import Enum


class Species(str, Enum):
    """Enum for the species of the animals in the UCI Zoo dataset."""
    mammal = "mammal"
    bird = "bird"
    reptile = "reptile"
    fish = "fish"
    amphibian = "amphibian"
    insect = "insect"
    molusc = "molusc"

    @classmethod
    def from_str(cls, value: str) -> Species:
        return getattr(cls, value)


# fetch dataset
zoo = fetch_ucirepo(id=111)

# data (as pandas dataframes)
X = zoo.data.features
y = zoo.data.targets

# This is a utility that allows each row to be a Case instance,
# which simplifies access to column values using dot notation.
all_cases = create_cases_from_dataframe(X, name="Animal")

# The targets are the species of the animals
category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
targets = [Species.from_str(category_id_to_name[i]) for i in y.values.flatten()]

# Now that we are done with the data preparation, we can create and use the Ripple Down Rules classifier.
grdr = GeneralRDR()

# Fit the GRDR to the data
case_queries = [CaseQuery(case, 'species', type(target), True, _target=target)
                for case, target in zip(all_cases[:10], targets[:10])]
grdr.fit(case_queries)

# Classify a case
cat = grdr.classify(all_cases[50])['species']
assert cat == targets[50]