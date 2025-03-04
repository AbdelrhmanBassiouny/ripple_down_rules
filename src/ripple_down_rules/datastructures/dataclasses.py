from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Union, Sequence, Any, Dict, List, TYPE_CHECKING, Optional

from ripple_down_rules.datastructures.operator import Operator, Equal

if TYPE_CHECKING:
    from .attribute import Attribute
    from .case import Case


@dataclass
class Condition:
    """
    A condition is a constraint on an attribute that must be satisfied for a case to be classified.
    """

    def __init__(self, name: str, value: Any, operator: Operator):
        """
        Create a condition.

        :param name: The name of the attribute that the condition is applied to.
        :param value: The value of the constraint.
        :param operator: The operator to compare the value to other values.
        """
        self.name = name
        self.value = value
        self.operator = operator

    @classmethod
    def from_two_cases(cls, old_case: Case, new_case: Case) -> Dict[str, Condition]:
        attributes_dict = new_case - old_case
        return cls.from_attributes(attributes_dict.values())

    @classmethod
    def from_str(cls, rule_str: str) -> Condition:
        operator = Operator.parse_operators(rule_str)[0]
        return cls(operator.arg_names[0], operator.arg_names[1], operator)

    @classmethod
    def from_case(cls, case: Case, operator: Operator = Equal()) -> Dict[str, Condition]:
        return cls.from_attributes(case.attributes_list, operator)

    @classmethod
    def from_attributes(cls, attributes: List[Attribute], operator: Operator = Equal()) -> Dict[str, Condition]:
        return {a.name: cls.from_attribute(a, operator) for a in attributes}

    @classmethod
    def from_attribute(cls, attribute: Attribute, operator: Operator = Equal()) -> Condition:
        return cls(attribute.name, attribute.value, operator)

    def __call__(self, x: Case) -> bool:
        return self.operator(x[self.name].value, self.value)

    def __str__(self):
        return f"{self.name} {self.operator} {self.value}"

    def __repr__(self):
        return self.__str__()


@dataclass
class ObjectAttributeTarget:
    obj: Any
    """
    The object that the attribute belongs to.
    """
    attribute_name: str
    """
    The name of the attribute.
    """
    target_value: Any
    """
    The target value of the attribute.
    """
    relational_representation: Optional[str] = None
    """
    The representation of the target value in relational form.
    """

    def __init__(self, obj: Any, attribute_name: str, target_value: Any,
                 relational_representation: Optional[str] = None):
        self.obj = obj
        self.name = attribute_name
        self.__class__.__name__ = self.name
        self.target_value = target_value
        self.relational_representation = relational_representation

    def __str__(self):
        if self.relational_representation:
            return f"{self.name} |= {self.relational_representation}"
        else:
            return f"{self.target_value}"


@dataclass
class Range:
    """
    A range is a pair of values that represents the minimum and maximum values of a numeric category.
    """
    min: Union[float, int]
    """
    The minimum value of the range.
    """
    max: Union[float, int]
    """
    The maximum value of the range.
    """
    min_closed: bool = True
    """
    Whether the minimum value is included in the range.
    """
    max_closed: bool = True
    """
    Whether the maximum value is included in the range.
    """

    def __contains__(self, item: Union[float, int, Sequence[Union[float, int]], Range]) -> bool:
        """
        Check if a value or an iterable of values are within the range.

        :param item: The value or values to check.
        """
        if not self.is_numeric(item):
            raise ValueError(f"Item {item} contains non-numeric values.")
        elif hasattr(item, "__iter__"):
            return min(item) in self and max(item) in self
        elif isinstance(item, Range):
            return self == item
        else:
            return self.is_numeric_value_in_range(item)

    def is_numeric_value_in_range(self, value: Union[float, int]) -> bool:
        """
        Check if a numeric value is in the range.

        :param value: The value to check.
        """
        satisfies_min = (self.min_closed and value >= self.min) or (not self.min_closed and value > self.min)
        satisfies_max = (self.max_closed and value <= self.max) or (not self.max_closed and value < self.max)
        return satisfies_min and satisfies_max

    @staticmethod
    def is_numeric(value: Any) -> bool:
        """
        Check if a value is numeric.

        :param value: The value to check.
        """
        if isinstance(value, str):
            return False
        elif hasattr(value, "__iter__"):
            return all(isinstance(i, (float, int)) for i in value)
        elif isinstance(value, Range):
            return value.is_numeric(value.min) and value.is_numeric(value.max)
        else:
            return isinstance(value, (float, int))

    def __eq__(self, other: Range) -> bool:
        if not isinstance(other, Range):
            return False
        return (self.min == other.min and self.max == other.max
                and self.min_closed == other.min_closed
                and self.max_closed == other.max_closed)

    def __hash__(self) -> int:
        return hash((self.min, self.max, self.min_closed, self.max_closed))

    def __str__(self) -> str:
        left = "[" if self.min_closed else "("
        right = "]" if self.max_closed else ")"
        return f"{left}{self.min}, {self.max}{right}"

    def __repr__(self) -> str:
        return self.__str__()
