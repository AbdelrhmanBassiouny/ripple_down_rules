from __future__ import annotations

import os
from unittest import TestCase

from typing_extensions import List, Optional, Set, Any

from ripple_down_rules.datastructures import Case, ObjectAttributeTarget
from ripple_down_rules.utils import get_property_name


class PhysicalObject:
    def __init__(self, name: str, contained_objects: Optional[List[PhysicalObject]] = None):
        self.name: str = name
        self._contained_objects: List[PhysicalObject] = contained_objects or []

    @property
    def contained_objects(self) -> List[PhysicalObject]:
        return self._contained_objects

    @contained_objects.setter
    def contained_objects(self, value: List[PhysicalObject]):
        self._contained_objects = value

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Part(PhysicalObject):
    ...


class Robot(PhysicalObject):
    parts: List[Part] = None
    """
    The robot parts.
    """
    def __init__(self, name: str, parts: Optional[List[Part]] = None):
        super().__init__(name)
        self.parts: List[Part] = parts if parts else []


class RelationalRDRTestCase(TestCase):
    case: Any
    target: Any
    test_results_dir: str = "./test_results"
    expert_answers_dir: str = "./test_expert_answers"
    robot: Robot

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.test_results_dir):
            os.makedirs(cls.test_results_dir)
        part_a = Part(name="A")
        part_b = Part(name="B")
        part_c = Part(name="C")
        part_d = Part(name="D")
        part_e = Part(name="E")
        robot = Robot("pr2", parts=[part_a, part_b, part_c, part_d])
        part_a.contained_objects = {part_b, part_c}
        part_c.contained_objects = {part_d}
        part_d.contained_objects = {part_e}
        cls.robot: Robot = robot
        cls.case = Case.from_object(robot)
        attr_name = get_property_name(robot, robot.contained_objects)
        cls.target = ObjectAttributeTarget(robot, attr_name, {part_b, part_c, part_d, part_e})
