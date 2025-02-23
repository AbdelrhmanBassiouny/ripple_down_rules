import os
from unittest import TestCase

from typing_extensions import List, Optional, Set, Any, Type

from ripple_down_rules.datastructures import Case, Attribute, \
    CategoryValueType, RDRMode, Categorical
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR


class PhysicalObject:
    def __init__(self, name: str):
        self.name = name
        self._contained_objects: Set[PhysicalObject] = set()

    @property
    def contained_objects(self):
        return self._contained_objects

    @contained_objects.setter
    def contained_objects(self, value):
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
        self.parts = parts or []


class TestRDR(TestCase):
    case: Case
    target: Type[Attribute]
    test_results_dir: str = "./test_results"
    expert_answers_dir: str = "./test_expert_answers"

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
        cls.case: Case = Case.from_object(robot)
        cls.target = type(cls.case["contained_objects"])([part_b, part_c, part_d, part_e])
        print(cls.target)

    def test_target(self):
        pass

    def test_classify_scrdr(self):
        use_loaded_answers = False
        save_answers = True
        filename = self.expert_answers_dir + "/relational_scrdr_expert_answers_classify"
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        scrdr = SingleClassRDR(mode=RDRMode.Relational)
        cat = scrdr.fit_case(self.case, self.target, expert=expert)
        self.assertEqual(cat, self.target)

        if save_answers:
            cwd = os.getcwd()
            file = os.path.join(cwd, filename)
            expert.save_answers(file)
