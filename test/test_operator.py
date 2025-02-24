from ripple_down_rules.datastructures import *
from unittest import TestCase



class TestOperator(TestCase):
    rule_str: str

    @classmethod
    def setUpClass(cls):
        cls.rule_str = "hasattr(A, a) and not b or c & d > e"

    def test_parse_operators(self):
        operators = Operator.parse_operators(self.rule_str)
        operator_types = [type(op) for op in operators]
        assert len(operators) == 6
        assert HasAttribute in operator_types
        assert Not in operator_types
        assert And in operator_types
        assert Or in operator_types
        assert Greater in operator_types
        assert BitAnd in operator_types
