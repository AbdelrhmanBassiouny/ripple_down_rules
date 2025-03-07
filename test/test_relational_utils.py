from test_relational_rdr import PhysicalObject, Part, Robot, RelationalRDRTestCase
from ripple_down_rules.utils import get_attribute_name_from_value


class RelationalUtilsTestCase(RelationalRDRTestCase):
    def test_get_property_name(self):
        self.assertEqual(get_attribute_name_from_value(self.case, self.case.contained_objects), "contained_objects")
