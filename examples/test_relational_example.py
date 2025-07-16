import pytest

from examples.relational_model import Part, Robot, PhysicalObject
from ripple_down_rules import GeneralRDR, CaseQuery
from ripple_down_rules.helpers import enable_gui

enable_gui()

@pytest.fixture
def robot_rdr():
    return GeneralRDR(save_dir='./', model_name='robot_rdr')

def robot_factory():
    part_a = Part(name="A")
    part_b = Part(name="B")
    part_c = Part(name="C")
    robot = Robot("pr2", parts=[part_a])
    part_a.contained_objects = [part_b]
    part_b.contained_objects = [part_c]
    return robot

def test_robot_contained_objects(robot_rdr):
    robot = robot_factory()
    cq = CaseQuery(robot, "contained_objects", (PhysicalObject,), False,
                   scenario=test_robot_contained_objects, case_factory=robot_factory)
    robot_rdr.fit_case(cq, update_existing_rules=False)
    result = robot_rdr.classify(robot)
    assert list(result['contained_objects']) == [robot.parts[0].contained_objects[0]]  # part_b