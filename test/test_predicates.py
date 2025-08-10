import pytest

from ripple_down_rules import *
from .datasets import Drawer, Handle


@pytest.fixture
def drawer_cabinet_dependency_graph():
    TrackedObjectMixin.make_class_dependency_graph(composition=True)


def test_fit_depends_on_predicate(drawer_cabinet_dependency_graph) -> None:
    dependsOn.rdr_decorator.fit = True
    dependsOn.rdr_decorator.update_existing_rules = False
    assert dependsOn(Drawer, Handle)
