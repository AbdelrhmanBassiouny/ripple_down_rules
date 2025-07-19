from typing_extensions import List, Set, Union
from relational_model import PhysicalObject, Robot


def conditions_298609776593271728826836208156881692889(case) -> bool:
    def conditions_for_robot_contained_objects_of_type_physical_object(case: Robot) -> bool:
        """Get conditions on whether it's possible to conclude a value for Robot.contained_objects  of type PhysicalObject."""
        return len(case.parts) > 0
    return conditions_for_robot_contained_objects_of_type_physical_object(case)


def conclusion_298609776593271728826836208156881692889(case) -> List[PhysicalObject]:
    def robot_contained_objects_of_type_physical_object(case: Robot) -> List[PhysicalObject]:
        """Get possible value(s) for Robot.contained_objects  of type PhysicalObject."""
        contained_objects: List[PhysicalObject] = []
        for part in case.parts:
            contained_objects.extend(part.contained_objects)
        return contained_objects
    return robot_contained_objects_of_type_physical_object(case)


