import os
from unittest import TestCase

from ripple_down_rules.utils import extract_imports, make_set


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
