from enum import Enum
from ripple_down_rules import datasets
from ormatic.ormatic import ORMatic
from ormatic.dao import AlternativeMapping
from ormatic.utils import recursive_subclasses, classes_of_module
from dataclasses import is_dataclass


def main():
    # get classes that should be mapped
    classes = set(recursive_subclasses(AlternativeMapping))
    classes |= set(classes_of_module(datasets))

    # remove classes that should not be mapped
    classes -= set(recursive_subclasses(Enum))
    classes -= set([cls for cls in classes if not is_dataclass(cls)])
    classes -= {datasets.MappedAnimal, datasets.HabitatTable, datasets.Habitat}
    ormatic = ORMatic(classes)
    ormatic.make_all_tables()

    with open('../../src/ripple_down_rules/orm_interface.py', 'w') as f:
        ormatic.to_sqlalchemy_file(f)


if __name__ == '__main__':
    main()