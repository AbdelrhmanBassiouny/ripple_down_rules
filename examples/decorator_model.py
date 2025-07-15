from dataclasses import dataclass

from typing_extensions import List

from relational_model import Part, Robot as OGRobot, PhysicalObject
from ripple_down_rules import RDRDecorator


@dataclass(unsafe_hash=True)
class Robot(OGRobot):
    containment_rdr: RDRDecorator = RDRDecorator("./", (PhysicalObject,), False,
                                                 fit=False)

    @containment_rdr.decorator
    def get_contained_objects(self) -> List[PhysicalObject]:
        """
        Returns the contained objects of the robot.
        """
        ...

def my_robot_factory() -> Robot:
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