import pytest

from ripple_down_rules.rdr import GeneralRDR
from datasets import Drawer, Handle, Cabinet


@pytest.mark.skip("Not Implemented yet")
def test_construct_class_dependency_graph():
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
