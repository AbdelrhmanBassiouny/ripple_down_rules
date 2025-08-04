import pytest
from typing_extensions import Iterable, Union

from ripple_down_rules.entity import an, entity
from ripple_down_rules.query import Generate, where
from ripple_down_rules import symbolic
from ripple_down_rules.symbolic import contains, in_, And
from .datasets import Handle, Body


def test_generate_with_using_attribute_and_callables(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world
    def generate_handles():
        with symbolic.SymbolicMode():
            body = an(Body, domain=world.bodies)
            yield from entity(body, body.name.startswith("Handle"))
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
            body = an(Body, domain=world.bodies)
            yield from entity(body, contains(body.name, "Handle"))
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
            body = an(Body, domain=world.bodies)
            yield from entity(body, in_("Handle", body.name))
    handles = list(generate_handles())
    assert len(handles) == 2, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_and(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world
    def generate_handles():
        with symbolic.SymbolicMode():
            body = an(Body, domain=world.bodies)
            yield from entity(body, contains(body.name, "Handle") & contains(body.name, '1'))
    handles = list(generate_handles())
    assert len(handles) == 1, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_or(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world
    def generate_handles():
        with symbolic.SymbolicMode():
            body = an(Body, domain=world.bodies)
            yield from entity(body,contains(body.name, "Handle1") | contains(body.name, 'Handle2'))
    handles = list(generate_handles())
    assert len(handles) == 2, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_multi_or(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world
    def generate_handles_and_container1():
        with symbolic.SymbolicMode():
            body = an(Body, domain=world.bodies)
            yield from entity(body,contains(body.name, "Handle1") | contains(body.name, 'Handle2')
                              | contains(body.name, 'Container1'))
    handles_and_container1 = list(generate_handles_and_container1())
    assert len(handles_and_container1) == 3, "Should generate at least one handle."
    # assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."

def test_generate_with_and_or(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_handles_and_container1():
        with symbolic.SymbolicMode():
            body = an(Body, domain=world.bodies)
            yield from entity(body, (contains(body.name, "Handle") & contains(body.name, '1'))
                              | (contains(body.name, 'Container') & contains(body.name, '1')))

    handles_and_container1 = list(generate_handles_and_container1())
    assert len(handles_and_container1) == 2, "Should generate at least one handle."