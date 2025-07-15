from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List


@dataclass(unsafe_hash=True)
class PhysicalObject:
    """
    A physical object is an object that can be contained in a container.
    """
    name: str
    contained_objects: List[PhysicalObject] = field(default_factory=list, hash=False)

@dataclass(unsafe_hash=True)
class Part(PhysicalObject):
    ...

@dataclass(unsafe_hash=True)
class Robot(PhysicalObject):
    parts: List[Part] = field(default_factory=list, hash=False)


def my_robot_factory()-> Robot:
    """
    Factory function to create a simple robot with parts and containment relationships.
    """
    part_a = Part(name="A")
    part_b = Part(name="B")
    part_c = Part(name="C")
    robot = Robot("pr2", parts=[part_a])
    part_a.contained_objects = [part_b]
    part_b.contained_objects = [part_c]

    return robot