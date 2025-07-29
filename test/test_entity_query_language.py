import pytest
from typing_extensions import Iterable, Union

from ripple_down_rules.entity import an, entity
from ripple_down_rules.query import Generate, where
from ripple_down_rules import symbolic
from ripple_down_rules.symbolic import contains, in_
from .datasets import Handle, Body


def test_generate_with_using_attribute_and_callables(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world
    def generate_handles():
        with symbolic.SymbolicMode():
            body = entity("Handle", an(Body, from_=world.bodies))
            where(body.name.startswith("Handle"))
            yield from body
    handles = list(generate_handles())
    assert len(handles) == 2, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_contains(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world
    def generate_handles():
        with symbolic.SymbolicMode():
            body = entity("Handle", an(Body, from_=world.bodies))
            where(contains(body.name, "Handle"))
            yield from Handle(body)
    handles = list(generate_handles())
    assert len(handles) == 2, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_in(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world
    def generate_handles():
        with symbolic.SymbolicMode():
            body = entity("Handle", an(Body, from_=world.bodies))
            where(in_("Handle", body.name))
            yield from Handle(body)
    handles = list(generate_handles())
    assert len(handles) == 2, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."

@pytest.mark.skip(reason="This test is not implemented yet.")
def test_generate_with_using_and(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world
    def generate_handles():
        with symbolic.SymbolicMode():
            body = entity("Handle", an(Body, from_=world.bodies))
            where(
                contains(body.name, "Handle") and contains(body.name, '1')
            )
            yield from Handle(body)
    handles = list(generate_handles())
    assert len(handles) == 2, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."