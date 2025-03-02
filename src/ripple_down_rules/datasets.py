import os
import pickle

from typing_extensions import Tuple, List
from ucimlrepo import fetch_ucirepo

from ripple_down_rules.datastructures import Case, Attribute, Species


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


def get_dataset(dataset_id, cache_file):
    """Fetches dataset from cache or downloads it if not available."""
    dataset = load_cached_dataset(cache_file)
    if dataset is None:
        print("Downloading dataset...")
        dataset = fetch_ucirepo(id=dataset_id)

        # Check if dataset is valid before caching
        if dataset is None or not hasattr(dataset, "data"):
            print("Error: Failed to fetch dataset.")
            return None

        save_dataset_to_cache(dataset, cache_file)

    return dataset


def load_zoo_dataset(cache_file: str) -> Tuple[List[Case], List[Attribute]]:
    """
    Load the zoo dataset.

    :param cache_file: the cache file.
    :return: all cases and targets.
    """
    # fetch dataset
    zoo = get_dataset(111, cache_file)

    # data (as pandas dataframes)
    X = zoo['features']
    y = zoo['targets']
    # get ids as list of strings
    ids = zoo['ids'].values.flatten()
    all_cases = Case.create_cases_from_dataframe(X, ids)

    category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
    category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
    targets = [Species(Species.Values.from_str(category_id_to_name[i])) for i in y.values.flatten()]
    return all_cases, targets
