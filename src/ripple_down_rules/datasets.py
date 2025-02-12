from enum import IntEnum

import numpy as np
import pandas as pd
from random_events.set import Set
from random_events.variable import Variable, Continuous, Integer, Symbolic
from typing_extensions import Tuple, List
from ucimlrepo import fetch_ucirepo

from ripple_down_rules.datastructures import Case, Category, Species
from ripple_down_rules.helpers import create_cases_from_dataframe


def load_zoo_dataset() -> Tuple[List[Case], List[Category]]:
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
    all_cases = create_cases_from_dataframe(X, ids)
    # print category names
    category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]

    category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
    targets = [Species(category_id_to_name[i]) for i in y.values.flatten()]
    return all_cases, targets


def infer_variables_from_dataframe(data: pd.DataFrame) \
        -> List[Variable]:
    """
    Infer the variables from a dataframe.
    The variables are inferred by the column names and types of the dataframe.

    :param data: The dataframe to infer the variables from.
    :return: The inferred variables.
    """
    result = []

    for column, datatype in zip(data.columns, data.dtypes):
        unique_values = data[column].unique()
        unique_values.sort()

        # handle continuous variables
        if datatype in [float]:
            variable = Continuous(column)

        # handle discrete variables
        elif datatype == int:
            variable = Integer(column)

        elif datatype in [bool, object]:
            all_elements = set(unique_values)
            variable = Symbolic(column, Set.from_iterable(all_elements))
        else:
            raise ValueError(f"Datatype {datatype} of column {column} is not supported.")
        result.append(variable)

    return result
