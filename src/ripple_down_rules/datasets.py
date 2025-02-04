from typing_extensions import Tuple, List
from ucimlrepo import fetch_ucirepo

from ripple_down_rules.datastructures import Case
from ripple_down_rules.helpers import create_cases_from_dataframe


def load_zoo_dataset() -> Tuple[List[Case], List[str]]:
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
    targets = [category_id_to_name[i] for i in y.values.flatten()]
    return all_cases, targets
