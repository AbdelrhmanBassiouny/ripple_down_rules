import os
from textwrap import dedent

from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.datastructures.enums import PromptFor
from ripple_down_rules.user_interface.template_file_creator import TemplateFileCreator
from test_rdr_world import World, Handle, Container



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

def test_load():
    # Test the load function
    world = World()
    imports = "from test_rdr_world import World\n\n\n"
    func_code = "def test_func(case):\n    return case"
    source_code = f"{imports}{func_code}\n"
    source_code = dedent(source_code)
    with open("test.py", "w") as f:
        f.write(source_code)
    code_lines, updates = TemplateFileCreator.load("test.py", "test_func")
    assert code_lines == func_code.splitlines()
    assert list(updates.keys()) == ["test_func"]
    assert updates["test_func"](world) == world
    os.remove("test.py")
