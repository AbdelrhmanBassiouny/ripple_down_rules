from typing_extensions import Iterable, Union

from ripple_down_rules.query import Generate
from ripple_down_rules.symbolic_variable import SymbolicMode
from .datasets import Handle, Body


def test_body_generator(handles_and_containers_world):
    world = handles_and_containers_world
    with SymbolicMode():
        bodies = Body(world.bodies)
    assert [b is wb for b, wb in zip(bodies, world.bodies)], "Body generator should yield all bodies in the world."


def test_generate_handles(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world
    def generate_handles():
        with SymbolicMode():
            bodies = Body(world.bodies)
            yield from Generate(Handle(bodies)).where(bodies.name.startswith("H"))
    handles = list(generate_handles())
    assert len(handles) == 2, "Should generate at least one handle."
    print(handles)