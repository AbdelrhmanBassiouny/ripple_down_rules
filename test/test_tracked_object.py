import os
from os.path import dirname

from ripple_down_rules import *
from .datasets import Drawer, Handle, Cabinet, View, WorldEntity, Body, Connection


def test_construct_class_hierarchy():
    TrackedObjectMixin._reset_dependency_graph()
    TrackedObjectMixin.make_class_dependency_graph(composition=False)
    Drawer.to_dot(os.path.join(dirname(__file__), "dependency_graph"))
    assert len(Drawer._dependency_graph.nodes()) == 20
    assert len(Drawer._dependency_graph.edges()) == 17


def test_construct_class_composition():
    TrackedObjectMixin._reset_dependency_graph()
    TrackedObjectMixin.make_class_dependency_graph(composition=True)
    Drawer.to_dot(os.path.join(dirname(__file__), "dependency_graph"))
    assert len(Drawer._dependency_graph.nodes()) == 20
    assert len(Drawer._dependency_graph.edges()) == 22
    Drawer.to_dot(os.path.join(dirname(__file__), "dependency_graph"))


# @pytest.mark.skip("Not Implemented yet")
def test_construct_class_composition_and_dependency():
    TrackedObjectMixin._reset_dependency_graph()
    TrackedObjectMixin.make_class_dependency_graph(composition=True)
    assert has(Drawer, Handle)
    assert has(Cabinet, Drawer)
    assert isA(Cabinet, View)
    assert isA(Cabinet, WorldEntity)
    assert not has(Cabinet, Handle)
    assert has(Cabinet, Handle, recursive=True)
    assert has(Cabinet, Body)
    assert has(Cabinet, WorldEntity)
    assert not has(Cabinet, Connection, recursive=True)
