from unittest import TestCase

from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.datastructures.enums import PromptFor
from ripple_down_rules.user_interface.template_file_creator import TemplateFileCreator
from test_rdr_world import World, Handle, Container


class TestTemplateFileCreator(TestCase):

    def test_func_name_with_one_type(self):
        # Test the function name
        world = World()
        case_query: CaseQuery = CaseQuery(world, "views", (Handle,), False)

        func_name = TemplateFileCreator.get_func_name(PromptFor.Conclusion, case_query)
        self.assertEqual(func_name, "world_views_of_type_handle")

        func_name = TemplateFileCreator.get_func_name(PromptFor.Conditions, case_query)
        self.assertEqual(func_name, "conditions_for_world_views_of_type_handle")

    def test_func_name_with_two_type(self):
        # Test the function name
        world = World()
        case_query: CaseQuery = CaseQuery(world, "views", (Handle, Container), False)

        func_name = TemplateFileCreator.get_func_name(PromptFor.Conclusion, case_query)
        self.assertEqual(func_name, "world_views_of_type_handle_or_container")

        func_name = TemplateFileCreator.get_func_name(PromptFor.Conditions, case_query)
        self.assertEqual(func_name, "conditions_for_world_views_of_type_handle_or_container")

    def test_func_name_with_not_needed_types(self):
        # Test the function name
        world = World()
        case_query: CaseQuery = CaseQuery(world, "views", (Handle, list, bool, type(None)), False)

        func_name = TemplateFileCreator.get_func_name(PromptFor.Conclusion, case_query)
        self.assertEqual(func_name, "world_views_of_type_handle")

        func_name = TemplateFileCreator.get_func_name(PromptFor.Conditions, case_query)
        self.assertEqual(func_name, "conditions_for_world_views_of_type_handle")
