from __main__ import Robot
from __main__ import PhysicalObject
from typing import List
from typing_extensions import Union


def conditions_292338421353147015053324365450552788888(case):
    def conditions_for_robot_contained_objects_of_type_physical_object(case: Robot) -> bool:
        """Get conditions on whether it's possible to conclude a value for Robot.contained_objects  of type PhysicalObject."""
        return len(case.parts) > 0
    return conditions_for_robot_contained_objects_of_type_physical_object(case)


def conclusion_292338421353147015053324365450552788888(case):
    def robot_contained_objects_of_type_physical_object(case: Robot) -> List[PhysicalObject]:
        """Get possible value(s) for Robot.contained_objects  of type PhysicalObject."""
        contained_objects = []
        for part in case.parts:
            contained_objects.extend(part.contained_objects)
        return contained_objects
    return robot_contained_objects_of_type_physical_object(case)


