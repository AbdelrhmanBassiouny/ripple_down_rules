from __future__ import annotations

import re
from abc import ABC, abstractmethod

from typing_extensions import Any, List, Optional, deprecated, Tuple

from ripple_down_rules.failures import InvalidOperator
from ripple_down_rules.utils import get_all_subclasses


@deprecated("This module is deprecated, use Operator.parse_operators instead.")
def str_to_operator_fn(rule_str: str) -> Tuple[Optional[str], Optional[str], Optional[Operator]]:
    """
    Convert a string containing a rule to a function that represents the rule.

    :param rule_str: A string that contains the rule.
    :return: An operator object and two arguments that represents the rule.
    """
    operator: Optional[Operator] = None
    arg1: Optional[str] = None
    arg2: Optional[str] = None
    operators = [LessEqual(), GreaterEqual(), Equal(), Less(), Greater(), In()]
    for op in operators:
        if op.__str__() in rule_str:
            operator = op
            break
    if not operator:
        raise InvalidOperator(rule_str, operators)
    if operator is not None:
        arg1, arg2 = rule_str.split(operator.__str__())
        arg1 = arg1.strip()
        arg2 = arg2.strip()
    return arg1, arg2, operator


class Operator(ABC):
    """
    An operator is a function that compares two values and returns a boolean value.
    """
    name: str
    arg_names: List[str]

    def __init__(self, *arg_names: List[str]):
        self.arg_names = arg_names

    @classmethod
    def parse_operators(cls, rule_str: str) -> List[Operator]:
        """
        Parse all operators in a rule string.

        :param rule_str: A string that contains the rule made up of operators.
        :return: A list of all operators in the rule string.
        """
        all_operators: List[cls] = []
        possible_operators = get_all_subclasses(cls)
        for op in possible_operators.values():
            if not hasattr(op, "name"):
                continue
            all_op_instances = re.findall(op.pattern(), rule_str)
            for instance in all_op_instances:
                all_operators.append(op.from_str(instance))
        return all_operators if len(all_operators) > 0 else None

    @classmethod
    @abstractmethod
    def pattern(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def from_str(cls, rule_str: str) -> Operator:
        """
        Convert a string containing a rule to an Operator Object.

        :param rule_str: A string that contains the rule.
        :return: An operator object that represents the rule.
        """
        pass

    @abstractmethod
    def __call__(self, *args) -> bool:
        pass

    @classmethod
    def __str__(cls) -> str:
        return cls.name

    def __repr__(self):
        return self.__str__()


class UnaryOperator(Operator, ABC):
    """
    A unary operator is an operator that compares one value with another value.
    """

    @abstractmethod
    def __call__(self, x: Any) -> bool:
        pass

    @classmethod
    def from_str(cls, rule_str: str) -> Operator:
        arg_name = re.findall(cls.pattern(), rule_str.strip())[0].strip()
        arg_name = arg_name.replace(cls.name, "").split(" ")[0]
        return cls(arg_name)

    @classmethod
    def pattern(cls) -> str:
        return rf"\s?{cls.name}.*"


class Not(UnaryOperator):
    """
    The not operator that checks if the value is false.
    """
    name: str = "not "

    def __call__(self, x: Any) -> bool:
        return not x


class BitNot(UnaryOperator):
    """
    The bitwise not operator that checks if the value is false.
    """
    name: str = "~"

    def __call__(self, x: Any) -> bool:
        return ~x


class Predicate(Operator, ABC):
    """
    A predicate is an operator that is represented by the regular expression "predicate(.*, .*,...)",
    it is a function that takes in one or multiple arguments and returns a boolean value.
    """

    @classmethod
    def from_str(cls, rule_str: str) -> Operator:
        arg_names = re.findall(cls.pattern(), rule_str.strip())[0].strip()
        arg_names = arg_names.replace(f"{cls.name}(", "").replace(")", "")
        arg_names = arg_names.split(",")
        arg_names = [arg.strip() for arg in arg_names]
        return cls(*arg_names)

    @classmethod
    def pattern(cls) -> str:
        return rf"\s?{cls.name}\(.*\)"


class Length(Predicate):
    """
    The length operator that checks if the length of the first value is equal to the second value.
    """
    name: str = "len"

    def __call__(self, x: Any) -> int:
        return len(x)


class HasAttribute(Predicate):
    """
    The has attribute operator that checks if the first value has the second value as an attribute.
    """
    name: str = "hasattr"

    def __call__(self, x: Any, y: Any) -> bool:
        return hasattr(x, y)


class IsInstance(Predicate):
    """
    The is instance operator that checks if the first value is an instance of the second value.
    """

    name: str = "isinstance"

    def __call__(self, x: Any, y: Any) -> bool:
        return isinstance(x, y)


class IsSubclass(Predicate):
    """
    The is subclass operator that checks if the first value is a subclass of the second value.
    """

    name: str = "issubclass"

    def __call__(self, x: Any, y: Any) -> bool:
        return issubclass(x, y)


class BinaryOperator(Operator, ABC):
    """
    A binary operator is an operator that compares two values, and is represented by the regular expression ".* op .*".
    """

    @abstractmethod
    def __call__(self, x: Any, y: Any) -> bool:
        pass

    @classmethod
    def from_str(cls, rule_str: str) -> Operator:
        arg_names = re.findall(cls.pattern(), rule_str)[0].strip()
        arg_names = arg_names.split(cls.name)
        arg_names = [arg.strip() for arg in arg_names]
        return cls(*arg_names)

    @classmethod
    def pattern(cls) -> str:
        return rf".*{cls.name}.*"


class And(BinaryOperator):
    """
    The and operator that checks if both values are true.
    """
    name: str = " and "

    def __call__(self, x: Any, y: Any) -> bool:
        return x and y


class BitAnd(BinaryOperator):
    """
    The bitwise and operator that checks if both values are true.
    """

    name: str = " & "

    def __call__(self, x: Any, y: Any) -> bool:
        return x & y


class Or(BinaryOperator):
    """
    The or operator that checks if one of the values is true.
    """
    name: str = " or "

    def __call__(self, x: Any, y: Any) -> bool:
        return x or y


class BitOr(BinaryOperator):
    """
    The bitwise or operator that checks if one of the values is true.
    """

    name: str = " \| "

    def __call__(self, x: Any, y: Any) -> bool:
        return x | y


class In(BinaryOperator):
    """
    The in operator that checks if the first value is in the second value.
    """
    name: str = " in "

    def __call__(self, x: Any, y: Any) -> bool:
        return x in y


class Equal(BinaryOperator):
    """
    An equal operator that checks if two values are equal.
    """
    name: str = "=="

    def __call__(self, x: Any, y: Any) -> bool:
        return x == y


class NotEqual(BinaryOperator):
    """
    A not equal operator that checks if two values are not equal.
    """
    name: str = "!="

    def __call__(self, x: Any, y: Any) -> bool:
        return x != y


class Greater(BinaryOperator):
    """
    A greater operator that checks if the first value is greater than the second value.
    """

    name: str = ">"

    def __call__(self, x: Any, y: Any) -> bool:
        return x > y


class GreaterEqual(BinaryOperator):
    """
    A greater or equal operator that checks if the first value is greater or equal to the second value.
    """

    name: str = ">="

    def __call__(self, x: Any, y: Any) -> bool:
        return x >= y


class Less(BinaryOperator):
    """
    A less operator that checks if the first value is less than the second value.
    """

    name: str = "<"

    def __call__(self, x: Any, y: Any) -> bool:
        return x < y


class LessEqual(BinaryOperator):
    """
    A less or equal operator that checks if the first value is less or equal to the second value.
    """

    name: str = "<="

    def __call__(self, x: Any, y: Any) -> bool:
        return x <= y
