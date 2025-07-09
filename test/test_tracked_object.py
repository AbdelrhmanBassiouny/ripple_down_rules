import os
from os.path import dirname

import pytest

from ripple_down_rules.rdr import GeneralRDR
from .datasets import Drawer, Handle, Cabinet

def test_construct_class_hierarchy():
    Drawer.make_class_dependency_graph()
    assert len(Drawer._dependency_graph.nodes()) == 16
    assert len(Drawer._dependency_graph.edges()) == 14
    Drawer.to_dot(os.path.join(dirname(__file__), "dependency_graph"))

@pytest.mark.skip("Not Implemented yet")
def test_construct_class_composition_and_dependency():
    assert Drawer.has_one(Handle)
    assert Cabinet.has_many(Drawer)
    assert Cabinet.depends_on(Drawer)
    assert Cabinet.depends_on(Handle)


@pytest.mark.skip("Not Implemented yet")
def test_rule_dependency_graph(drawer_cabinet_rdr: GeneralRDR):
    drawer_rule = [r for r in [drawer_cabinet_rdr.start_rule] + list(drawer_cabinet_rdr.start_rule.descendants)
                   if Drawer in r.conclusion.conclusion_type][0]
    cabinet_rule = [r for r in [drawer_cabinet_rdr.start_rule] + list(drawer_cabinet_rdr.start_rule.descendants)
                    if Cabinet in r.conclusion.conclusion_type][0]
    assert cabinet_rule.depends_on(drawer_rule)
