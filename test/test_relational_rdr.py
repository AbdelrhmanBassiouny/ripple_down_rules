from __future__ import annotations

import os
from unittest import TestCase

from typing_extensions import List, Optional, Any

from ripple_down_rules.datastructures import CaseQuery, CallableExpression
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR
from ripple_down_rules.utils import render_tree


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

    def __init__(self, name: str, parts: Optional[List[Part]] = None):
        super().__init__(name)
        self.parts: List[Part] = parts if parts else []


class RelationalRDRTestCase(TestCase):
    case: Any
    target: Any
    test_results_dir: str = "./test_results"
    expert_answers_dir: str = "./test_expert_answers"
    robot: Robot
    part_a: Part
    part_b: Part
    part_c: Part
    part_d: Part
    part_e: Part
    part_f: Part

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.test_results_dir):
            os.makedirs(cls.test_results_dir)
        cls.part_a = Part(name="A")
        cls.part_b = Part(name="B")
        cls.part_c = Part(name="C")
        cls.part_d = Part(name="D")
        cls.part_e = Part(name="E")
        cls.part_f = Part(name="F")
        robot = Robot("pr2", parts=[cls.part_a, cls.part_b, cls.part_c, cls.part_d])
        cls.part_a.contained_objects = [cls.part_b, cls.part_c]
        cls.part_c.contained_objects = [cls.part_d]
        cls.part_d.contained_objects = [cls.part_e]
        cls.part_e.contained_objects = [cls.part_f]
        cls.robot: Robot = robot
        cls.target = CaseQuery(robot, attribute_name="contained_objects",
                               target=[cls.part_b, cls.part_c, cls.part_d, cls.part_e])

    def test_classify_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = self.expert_answers_dir + "/relational_scrdr_expert_answers_classify"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        scrdr = SingleClassRDR()
        cat = scrdr.fit_case(CaseQuery(self.robot, "contained_objects"), expert=expert)
        render_tree(scrdr.start_rule, use_dot_exporter=True,
                    filename=self.test_results_dir + "/relational_scrdr_classify")
        assert cat == self.target.target

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_parse_relational_conditions(self):
        user_input = "case.parts is not None and len(case.parts) > 0"
        conditions = CallableExpression(user_input, bool)
        print(conditions)
        print(conditions(self.robot))
        assert conditions(self.robot) == (self.robot.parts is not None and len(self.robot.parts) > 0)

    def test_parse_relational_conclusions(self):
        user_input = "case.parts.contained_objects"
        conclusion = CallableExpression(user_input, list)
        print(conclusion)
        print(conclusion(self.robot))
        assert conclusion(self.robot) == self.target.target
