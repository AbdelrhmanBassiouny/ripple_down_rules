import os
from os.path import dirname

import pytest
from rustworkx import PyDAG

from ripple_down_rules import *
from ripple_down_rules.rdr import GeneralRDR
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


@pytest.mark.skip("Not Implemented yet")
def test_rule_dependency_graph(drawer_cabinet_rdr: GeneralRDR):
    drawer_rule = [r for r in [drawer_cabinet_rdr.start_rule] + list(drawer_cabinet_rdr.start_rule.descendants)
                   if Drawer in r.conclusion.conclusion_type][0]
    cabinet_rule = [r for r in [drawer_cabinet_rdr.start_rule] + list(drawer_cabinet_rdr.start_rule.descendants)
                    if Cabinet in r.conclusion.conclusion_type][0]
    assert dependsOn(cabinet_rule, drawer_rule)
