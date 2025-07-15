import os
from os.path import dirname

import pytest
from rustworkx import PyDAG

from ripple_down_rules.rdr import GeneralRDR
from .datasets import Drawer, Handle, Cabinet, View, WorldEntity, Body, Connection


def test_construct_class_hierarchy():
    # Drawer._reset_dependency_graph()
    Drawer.make_class_dependency_graph(composition=False)
    Drawer.to_dot(os.path.join(dirname(__file__), "dependency_graph"))
    assert len(Drawer._dependency_graph.nodes()) == 16
    assert len(Drawer._dependency_graph.edges()) == 14


def test_construct_class_composition():
    # Drawer._reset_dependency_graph()
    Drawer.make_class_dependency_graph(composition=True)
    Drawer.to_dot(os.path.join(dirname(__file__), "dependency_graph"))
    assert len(Drawer._dependency_graph.nodes()) == 16
    assert len(Drawer._dependency_graph.edges()) == 19
    Drawer.to_dot(os.path.join(dirname(__file__), "dependency_graph"))


# @pytest.mark.skip("Not Implemented yet")
def test_construct_class_composition_and_dependency():
    # Drawer._reset_dependency_graph()
    Drawer.make_class_dependency_graph(composition=True)
    assert Drawer.has(Handle)
    assert Cabinet.has(Drawer)
    assert Cabinet.is_a(View)
    assert Cabinet.is_a(WorldEntity)
    assert not Cabinet.has(Handle)
    assert Cabinet.has(Handle, recursive=True)
    assert Cabinet.has(Body)
    assert Cabinet.has(WorldEntity)
    assert not Cabinet.has(Connection, recursive=True)


@pytest.mark.skip("Not Implemented yet")
def test_rule_dependency_graph(drawer_cabinet_rdr: GeneralRDR):
    drawer_rule = [r for r in [drawer_cabinet_rdr.start_rule] + list(drawer_cabinet_rdr.start_rule.descendants)
                   if Drawer in r.conclusion.conclusion_type][0]
    cabinet_rule = [r for r in [drawer_cabinet_rdr.start_rule] + list(drawer_cabinet_rdr.start_rule.descendants)
                    if Cabinet in r.conclusion.conclusion_type][0]
    assert cabinet_rule.depends_on(drawer_rule)
