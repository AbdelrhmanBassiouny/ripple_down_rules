import os
from textwrap import dedent

from typing_extensions import List

from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.datastructures.enums import PromptFor
from ripple_down_rules.rdr_decorators import RDRDecorator
from ripple_down_rules.user_interface.template_file_creator import TemplateFileCreator
from ripple_down_rules.utils import make_set
from .datasets import World, Handle, Container
from .datasets import Part, PhysicalObject, Robot



def test_func_name_with_one_type():
    # Test the function name
    world = World()
    case_query: CaseQuery = CaseQuery(world, "views", (Handle,), False)

    func_name = TemplateFileCreator.get_func_name(PromptFor.Conclusion, case_query)
    assert func_name == "world_views_of_type_handle"

    func_name = TemplateFileCreator.get_func_name(PromptFor.Conditions, case_query)
    assert func_name == "conditions_for_world_views_of_type_handle"

def test_func_name_with_two_type():
    # Test the function name
    world = World()
    case_query: CaseQuery = CaseQuery(world, "views", (Handle, Container), False)

    func_name = TemplateFileCreator.get_func_name(PromptFor.Conclusion, case_query)
    assert func_name == "world_views_of_type_handle_or_container"

    func_name = TemplateFileCreator.get_func_name(PromptFor.Conditions, case_query)
    assert func_name == "conditions_for_world_views_of_type_handle_or_container"

def test_func_name_with_not_needed_types():
    # Test the function name
    world = World()
    case_query: CaseQuery = CaseQuery(world, "views", (Handle, list, bool, type(None)), False)

    func_name = TemplateFileCreator.get_func_name(PromptFor.Conclusion, case_query)
    assert func_name == "world_views_of_type_handle"

    func_name = TemplateFileCreator.get_func_name(PromptFor.Conditions, case_query)
    assert func_name == "conditions_for_world_views_of_type_handle"

def test_rdr_decorator_func_name():
    class Example:
        def is_a_robot(self) -> bool:
            pass
        def select_objects_that_are_parts_of_robot(self, objects: List[PhysicalObject], robot: Robot) -> List[PhysicalObject]:
            pass
    example = Example()
    cq = RDRDecorator.create_case_query_from_method(example.is_a_robot, {"output_": None}, bool, True)
    func_name = TemplateFileCreator.get_func_name(PromptFor.Conclusion, cq)
    assert func_name == "example_is_a_robot"
    func_name = TemplateFileCreator.get_func_name(PromptFor.Conditions, cq)
    assert func_name == "conditions_for_example_is_a_robot"

    objects = [Part("Object1"), Part("Object2"), Part("Object3")]
    robot = Robot("Robot1", objects[:2])
    cq = RDRDecorator.create_case_query_from_method(
        example.select_objects_that_are_parts_of_robot, {"output_": None}, (List[PhysicalObject],),
        False, *(objects, robot))

    func_name = TemplateFileCreator.get_func_name(PromptFor.Conclusion, cq)
    assert func_name == "example_select_objects_that_are_parts_of_robot"
    func_name = TemplateFileCreator.get_func_name(PromptFor.Conditions, cq)
    assert func_name == "conditions_for_example_select_objects_that_are_parts_of_robot"

def test_load():
    # Test the load function
    world = World()
    imports = "from test.datasets import World\n\n\n"
    func_code = "def test_func(case):\n    return case"
    source_code = f"{imports}{func_code}\n"
    source_code = dedent(source_code)
    with open("test.py", "w") as f:
        f.write(source_code)
    code_lines, updates = TemplateFileCreator.load("test.py", "test_func")
    assert code_lines == func_code.splitlines()
    assert make_set(updates.keys()) == {"test_func", "World"}
    assert updates["test_func"](world) == world
    os.remove("test.py")
