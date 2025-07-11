import os
from os.path import dirname

import pytest
from typing_extensions import Type, Optional, Any, Callable

from ripple_down_rules.rdr_decorators import RDRDecorator, fit_rdr_func
from ripple_down_rules.datastructures.tracked_object import TrackedObjectMixin
from .datasets import Drawer, Handle, Cabinet


models_dir = os.path.join(dirname(__file__), "../src/ripple_down_rules/predicates_models")
depends_on_rdr: RDRDecorator = RDRDecorator(models_dir, (bool,), True,
                                     package_name='ripple_down_rules',
                                     fit=False)


@depends_on_rdr.decorator
def depends_on(parent_type: Type[TrackedObjectMixin], child_type: Type[TrackedObjectMixin]) -> bool:
    return False


@pytest.fixture
def drawer_cabinet_dependency_graph():
    Drawer.make_class_dependency_graph(composition=True)


def test_fit_depends_on_predicate(drawer_cabinet_dependency_graph) -> None:
    fit_rdr_func(test_fit_depends_on_predicate, depends_on, Drawer, Handle)
    assert depends_on(Drawer, Handle)
