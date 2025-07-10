import os
from os.path import dirname

import pytest
from typing_extensions import Type, Optional, Any, Callable

from ripple_down_rules.rdr_decorators import RDRDecorator, fit_rdr_func
from ripple_down_rules.datastructures.dataclasses import CaseFactoryMetaData
from ripple_down_rules.datastructures.tracked_object import TrackedObjectMixin
from .datasets import Drawer, Handle


models_dir = os.path.join(dirname(__file__), "../src/ripple_down_rules/predicates_models")
has_rdr: RDRDecorator = RDRDecorator(models_dir, (bool,), True,
                                     package_name='ripple_down_rules',
                                     fit=True)


@has_rdr.decorator
def has(parent_type: Type[TrackedObjectMixin], child_type: Type[TrackedObjectMixin]) -> bool:
    pass


@pytest.fixture
def drawer_cabinet_dependency_graph():
    Drawer.make_class_dependency_graph(composition=True)


def test_fit_has_predicate(drawer_cabinet_dependency_graph) -> None:
    fit_rdr_func(test_fit_has_predicate, has, True, Drawer, Handle)
