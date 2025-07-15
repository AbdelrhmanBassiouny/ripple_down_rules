import pytest

from .datasets import Drawer, Handle


@pytest.fixture
def drawer_cabinet_dependency_graph():
    Drawer.make_class_dependency_graph(composition=True)


def test_fit_depends_on_predicate(drawer_cabinet_dependency_graph) -> None:
    assert Drawer.depends_on(Handle)
