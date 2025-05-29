import os
from unittest import TestCase

from typing_extensions import List, Dict, Union, Tuple

from ripple_down_rules.utils import extract_imports, make_set, stringify_hint


class UtilsTestCase(TestCase):

    def test_extract_imports_from_file(self):
        # Test with a file that has imports
        file_path = "test_file.py"
        with open(file_path, "w") as f:
            f.write("import os\n")
            f.write("from module import function\n")
            f.write("from ripple_down_rules.utils import make_set\n")
            f.write("print('Hello World')\n")

        expected_scope = {"os": os, "make_set": make_set}
        actual_imports = extract_imports(file_path)
        self.assertEqual(expected_scope, actual_imports)

        # Clean up
        os.remove(file_path)


    def test_stringify_hint(self):

        self.assertEqual(stringify_hint(int), "int")
        self.assertEqual(stringify_hint(str), "str")
        self.assertEqual(stringify_hint(List[int]), "List[int]")
        self.assertEqual(stringify_hint(Dict[str, Union[int, str]]), "Dict[str, Union[int, str]]")
        self.assertEqual(stringify_hint(None), "None")
        self.assertEqual(stringify_hint("CustomType"), "CustomType")
        self.assertEqual(stringify_hint(List[Dict[str, Union[int, str]]]), "List[Dict[str, Union[int, str]]]")
        self.assertEqual(stringify_hint(List[Dict[str, Union[int, List[Tuple[str, float]]]]]),
                                        "List[Dict[str, Union[int, List[Tuple[str, float]]]]]")
