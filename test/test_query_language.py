import pytest
from typing_extensions import Iterable, Union

from ripple_down_rules.query import Generate
from ripple_down_rules import symbolic
from ripple_down_rules.symbolic import contains, in_
from .datasets import Handle, Body


def test_body_generator(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic.SymbolicMode():
        bodies = Body(world.bodies)
    assert [b is wb for b, wb in zip(bodies, world.bodies)], "Body generator should yield all bodies in the world."


def test_generate_with_using_attribute_and_callables(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world
    def generate_handles():
        with symbolic.SymbolicMode():
            bodies = Body(world.bodies)
            yield from Generate(Handle(bodies)).where(bodies.name.startswith("H"))
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
            bodies = Body(world.bodies)
            yield from Generate(Handle(bodies)).where(contains(bodies.name, "Handle"))
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
            bodies = Body(world.bodies)
            yield from Generate(Handle(bodies)).where(in_("Handle", bodies.name))
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
            bodies = Body(world.bodies)
            yield from Generate(Handle(bodies)).where(
                contains(bodies.name, "Handle") and contains(bodies.name, '1')
            )
    handles = list(generate_handles())
    assert len(handles) == 2, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."