import pytest
from typing_extensions import Iterable, Union

from ripple_down_rules.entity import an, entity
from ripple_down_rules.query import Generate, where
from ripple_down_rules import symbolic
from ripple_down_rules.symbolic import contains, in_, And
from .datasets import Handle, Body, Container, Drawer, FixedConnection, PrismaticConnection


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
    assert len(handles) == 3, "Should generate at least one handle."
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
    assert len(handles) == 3, "Should generate at least one handle."
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
    assert len(handles) == 3, "Should generate at least one handle."
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


def test_generate_with_multi_and(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_container1():
        with symbolic.SymbolicMode():
            body = an(Body, domain=world.bodies)
            yield from entity(body, contains(body.name, "n") & contains(body.name, '1')
                              & contains(body.name, 'C'))

    container1 = list(generate_container1())
    assert len(container1) == 1, "Should generate one container."
    assert isinstance(container1[0], Container), "The generated item should be of type Container."


def test_generate_with_more_than_one_source(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_drawers():
        with symbolic.SymbolicMode():
            container = an(Container, domain=world.bodies)
            handle = an(Handle, domain=world.bodies)
            fixed_connection = an(FixedConnection, domain=world.connections)
            prismatic_connection = an(PrismaticConnection, domain=world.connections)
            entity(fixed_connection,
                   (container == fixed_connection.parent) & (handle == fixed_connection.child) &
                   (container == prismatic_connection.child))
            yield from zip(container, handle, fixed_connection, prismatic_connection)
    for c, h, fc, pc in generate_drawers():
        assert c[1] == fc[1].parent
        assert h[1] == fc[1].child
        assert pc[1].child == fc[1].parent

    # handles_and_container1 = list(generate_drawers())
    # assert len(handles_and_container1) > 0, "Should generate at least one drawer."