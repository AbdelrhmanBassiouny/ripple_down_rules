from relational_rdr_test_case import PhysicalObject, Part, Robot, RelationalRDRTestCase
from ripple_down_rules.utils import get_property_name


class RelationalUtilsTestCase(RelationalRDRTestCase):
    def test_get_property_name(self):
        self.assertEqual(get_property_name(self.case, self.case.contained_objects), "contained_objects")
