from ripple_down_rules import *
from .datasets import Drawer, Handle


def test_fit_depends_on_predicate() -> None:
    dependsOn.rdr_decorator.fit = True
    dependsOn.rdr_decorator.update_existing_rules = False
    assert any(dependsOn(Drawer, Handle))
