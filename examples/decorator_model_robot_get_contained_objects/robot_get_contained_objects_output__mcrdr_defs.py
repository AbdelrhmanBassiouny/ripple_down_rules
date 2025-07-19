from ripple_down_rules.datastructures.case import Case
from typing_extensions import Dict, List, Set, Union
from decorator_model import Robot
from relational_model import PhysicalObject


def conditions_167615852950279355863004646114673699744(case) -> bool:
    def conditions_for_robot_get_contained_objects(self_: Robot, output_: PhysicalObject) -> bool:
        """Get conditions on whether it's possible to conclude a value for Robot_get_contained_objects.output_  of type PhysicalObject."""
        return len(self_.parts) > 0
    return conditions_for_robot_get_contained_objects(**case)


def conclusion_167615852950279355863004646114673699744(case) -> List[PhysicalObject]:
    def robot_get_contained_objects(self_: Robot, output_: PhysicalObject) -> List[PhysicalObject]:
        """Get possible value(s) for Robot_get_contained_objects.output_  of type PhysicalObject."""
        contained_objects = []
        for part in self_.parts:
            contained_objects.extend(part.contained_objects)
        return contained_objects
    return robot_get_contained_objects(**case)


