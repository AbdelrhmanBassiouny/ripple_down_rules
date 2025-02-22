from __future__ import annotations

import importlib.util
import os
from abc import abstractmethod, ABC
from collections import UserDict
from dataclasses import dataclass
from enum import Enum, auto

from orderedset import OrderedSet
from typing_extensions import Any, Callable, Tuple, Optional, List, Dict, Type, Union, Sequence

from .failures import InvalidOperator
from .utils import make_set, make_value_or_raise_error, import_class


class MCRDRMode(Enum):
    """
    The modes of the MultiClassRDR.
    """
    StopOnly = auto()
    """
    StopOnly mode, stop wrong conclusion from being made and does not add a new rule to make the correct conclusion.
    """
    StopPlusRule = auto()
    """
    StopPlusRule mode, stop wrong conclusion from being made and adds a new rule with same conditions as stopping rule
     to make the correct conclusion.
    """
    StopPlusRuleCombined = auto()
    """
    StopPlusRuleCombined mode, stop wrong conclusion from being made and adds a new rule with combined conditions of
    stopping rule and the rule that should have fired.
    """


class RDREdge(Enum):
    Refinement = "except if"
    """
    Refinement edge, the edge that represents the refinement of an incorrectly fired rule.
    """
    Alternative = "else if"
    """
    Alternative edge, the edge that represents the alternative to the rule that has not fired.
    """
    Next = "next"
    """
    Next edge, the edge that represents the next rule to be evaluated.
    """


class CategoryValueType(Enum):
    Unary = auto()
    """
    Unary value type (eg. null).
    """
    Binary = auto()
    """
    Binary value type (eg. True, False).
    """
    Discrete = auto()
    """
    Discrete value type (eg. 1, 2, 3).
    """
    Continuous = auto()
    """
    Continuous value type (eg. 1.0, 2.5, 3.4).
    """
    Nominal = auto()
    """
    Nominal value type (eg. red, blue, green), categories where the values have no natural order.
    """
    Ordinal = auto()
    """
    Ordinal value type (eg. low, medium, high), categories where the values have a natural order.
    """


class Attribute(ABC):
    """
    An attribute is a name-value pair that represents a feature of a case.
    an attribute can be used to compare two cases, to make a conclusion (which is also an attribute) about a case.
    """
    mutually_exclusive: bool = False
    """
    Whether the attribute is mutually exclusive, this means that the attribute instance can only have one value.
    """
    value_type: CategoryValueType = CategoryValueType.Nominal
    """
    The type of the value of the attribute.
    """
    _range: Union[set, Range]
    """
    The range of the attribute, this can be a set of possible values or a range of numeric values (int, float).
    """

    _registry: Dict[str, Type[Attribute]] = {}
    """
    A dictionary of all dynamically created subclasses of the attribute class.
    """

    @classmethod
    def register(cls, subclass: Type[Attribute]):
        """
        Register a subclass of the attribute class, this is used to be able to dynamically create Attribute subclasses.

        :param subclass: The subclass to register.
        """
        if not issubclass(subclass, Attribute):
            raise ValueError(f"{subclass} is not a subclass of Attribute.")
        # Add the subclass to the registry if it is not already in the registry.
        if subclass not in cls._registry:
            cls._registry[subclass.__name__.lower()] = subclass
        else:
            raise ValueError(f"{subclass} is already registered.")

    @classmethod
    def get_subclass(cls, name: str) -> Type[Attribute]:
        """
        Get a subclass of the attribute class by name.

        :param name: The name of the subclass.
        :return: The subclass.
        """
        if ' ' in name or '_' in name:
            name = name.capitalize().strip(' _')

        if name.lower() in cls._registry:
            return cls._registry[name.lower()]
        raise ValueError(f"No subclass with name {name}.")

    def __init_subclass__(cls, **kwargs):
        """
        Set the name of the attribute class to the name of the class in lowercase.
        """
        super().__init_subclass__(**kwargs)
        cls.name = cls.__name__.lower()

    def __init__(self, value: Any):
        """
        Create an attribute.

        :param value: The value of the attribute.
        """
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: Any):
        if not self.mutually_exclusive:
            self._value = make_set(value)
        elif self.mutually_exclusive:
            self._value = make_value_or_raise_error(value)

    def __eq__(self, other: Attribute):
        if not isinstance(other, Attribute):
            return False
        if self.name != other.name:
            return False
        if isinstance(self.value, set) and not isinstance(other.value, set):
            return self.value == make_set(other.value)
        else:
            return self.value == other.value

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if hasattr(value, "__iter__") and not isinstance(value, str):
            return set(value).issubset(cls._range)
        elif isinstance(value, str):
            return value.lower() in cls._range
        else:
            return value in cls._range

    @classmethod
    def is_within_range(cls, _range: Union[set, Range, Any]) \
            -> bool:
        if isinstance(cls._range, set):
            if hasattr(_range, "__iter__") and not isinstance(_range, str):
                return set(_range).issubset(cls._range)
            elif isinstance(_range, Range):
                return False
            elif isinstance(_range, str):
                return _range.lower() in cls._range
            else:
                return _range in cls._range
        else:
            return _range in cls._range

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f"{self.name}: {self.value}"

    def __repr__(self):
        return self.__str__()


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

    def __str__(self) -> str:
        left = "[" if self.min_closed else "("
        right = "]" if self.max_closed else ")"
        return f"{left}{self.min}, {self.max}{right}"

    def __repr__(self) -> str:
        return self.__str__()


class Integer(Attribute):
    """
    A discrete attribute is an attribute that has a value that is a discrete category.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Ordinal
    _range: Range = Range(-float("inf"), float("inf"), min_closed=False, max_closed=False)

    def __init__(self, value: Any):
        super().__init__(int(value))


class Continuous(Attribute):
    """
    A continuous attribute is an attribute that has a value that is a continuous category.
    """
    mutually_exclusive: bool = False
    value_type = CategoryValueType.Continuous
    _range: Range = Range(-float("inf"), float("inf"), min_closed=False, max_closed=False)

    def __init__(self, value: Any):
        super().__init__(float(value))


class CategoricalValue(Enum):
    """
    A categorical value is a value that is a category.
    """

    def __eq__(self, other):
        if isinstance(other, CategoricalValue):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return self.name == other

    def __hash__(self):
        return hash(self.name)

    @classmethod
    def to_list(cls):
        return list(cls._value2member_map_.values())

    @classmethod
    def from_str(cls, category: str):
        return cls[category.lower()]

    @classmethod
    def from_strs(cls, categories: List[str]):
        return [cls.from_str(c) for c in categories]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class Categorical(Attribute, ABC):
    """
    A categorical attribute is an attribute that has a value that is a category.
    """
    mutually_exclusive: bool = False
    value_type = CategoryValueType.Nominal
    _range: set
    Values: Type[CategoricalValue]

    def __init_subclass__(cls, **kwargs):
        """
        Create the Values enum class for the categorical attribute, this enum class contains all the possible values
        of the attribute.
        Note: This method is called when a subclass of Categorical is created (not when an instance is created).
        """
        super().__init_subclass__(**kwargs)
        cls.create_values()

    def __init__(self, value: Union[Categorical.Values, str]):
        super().__init__(value)
        if isinstance(value, str):
            self.value = self.Values[value.lower()]

    @classmethod
    def from_str(cls, category: str):
        return cls(cls.Values[category.capitalize()])

    @classmethod
    def from_strs(cls, categories: List[str]):
        return [cls.from_str(c) for c in categories]

    @classmethod
    def add_new_categories(cls, categories: List[str]):
        for category in categories:
            cls.add_new_category(category)

    @classmethod
    def add_new_category(cls, category: str):
        cls._range.add(category.lower())
        cls.create_values()

    @classmethod
    def create_values(cls):
        cls.Values = CategoricalValue(f"{cls.__name__}Values", {c.lower(): c.lower() for c in cls._range})


class Bool(Attribute):
    """
    A binary attribute is an attribute that has a value that is a binary category.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Binary
    _range: set = {True, False}

    def __init__(self, value: Any):
        super().__init__(bool(value))


class Unary(Attribute):
    """
    A unary attribute is an attribute that has a value that is a unary category.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Unary
    _range: set

    def __init__(self):
        super().__init__(self.__class__.__name__)


class Stop(Unary):
    """
    A stop category is a special category that represents the stopping of the classification to prevent a wrong
    conclusion from being made.
    """


class Species(Categorical):
    """
    A species category is a category that represents the species of an animal.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Nominal
    _range: set = {"mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"}

    def __init__(self, species: Union[Species.Values, str]):
        super().__init__(species)


class Habitat(Categorical):
    """
    A habitat category is a category that represents the habitat of an animal.
    """
    mutually_exclusive: bool = False
    value_type = CategoryValueType.Nominal
    _range: set = {"land", "water", "air"}

    def __init__(self, habitat: Union[Habitat.Values, str]):
        super().__init__(habitat)


class Operator(ABC):
    """
    An operator is a function that compares two values and returns a boolean value.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def __call__(self, x: Any, y: Any) -> bool:
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class In(Operator):
    """
    The in operator that checks if the first value is in the second value.
    """

    def __call__(self, x: Any, y: Any) -> bool:
        return x in y

    @property
    def name(self) -> str:
        return " in "


class Equal(Operator):
    """
    An equal operator that checks if two values are equal.
    """

    def __call__(self, x: Any, y: Any) -> bool:
        return x == y

    @property
    def name(self) -> str:
        return "=="


class Greater(Operator):
    """
    A greater operator that checks if the first value is greater than the second value.
    """

    def __call__(self, x: Any, y: Any) -> bool:
        return x > y

    @property
    def name(self) -> str:
        return ">"


class GreaterEqual(Operator):
    """
    A greater or equal operator that checks if the first value is greater or equal to the second value.
    """

    def __call__(self, x: Any, y: Any) -> bool:
        return x >= y

    @property
    def name(self) -> str:
        return ">="


class Less(Operator):
    """
    A less operator that checks if the first value is less than the second value.
    """

    def __call__(self, x: Any, y: Any) -> bool:
        return x < y

    @property
    def name(self) -> str:
        return "<"


class LessEqual(Operator):
    """
    A less or equal operator that checks if the first value is less or equal to the second value.
    """

    def __call__(self, x: Any, y: Any) -> bool:
        return x <= y

    @property
    def name(self) -> str:
        return "<="


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
        arg1, arg2, operator = str_to_operator_fn(rule_str)
        return cls(arg1, arg2, operator)

    @classmethod
    def from_case(cls, case: Case, operator: Operator = Equal()) -> Dict[str, Condition]:
        return cls.from_attributes(case.attributes_list, operator)

    @classmethod
    def from_attributes(cls, attributes: List[Attribute], operator: Operator = Equal()) -> Dict[str, Condition]:
        return {a.name: cls.from_attribute(a, operator) for a in attributes}

    @classmethod
    def from_attribute(cls, attribute: Attribute, operator: Operator = Equal()) -> Condition:
        return cls(attribute.name, attribute.value, operator)

    def __call__(self, x: Any) -> bool:
        return self.operator(x, self.value)

    def __str__(self):
        return f"{self.name} {self.operator} {self.value}"

    def __repr__(self):
        return self.__str__()


class Attributes(UserDict):
    """
    A collection of attributes that represents a set of constraints on a case. This is a dictionary where the keys are
    the names of the attributes and the values are the attributes. All are stored in lower case.
    """

    def __getitem__(self, item: Union[str, Attribute]) -> Attribute:
        if isinstance(item, Attribute):
            return self[item.name]
        else:
            return super().__getitem__(item.lower())

    def __setitem__(self, name: str, value: Attribute):
        name = name.lower()
        if name in self:
            if not value.mutually_exclusive:
                self[name].value.update(value)
            else:
                raise ValueError(f"Attribute {name} already exists in the case and is mutually exclusive.")
        else:
            super().__setitem__(name, value)

    def __contains__(self, item):
        return super().__contains__(item.lower())

    def __delitem__(self, key):
        super().__delitem__(key.lower())

    def __eq__(self, other):
        if not isinstance(other, (Attributes, dict)):
            return False
        elif isinstance(other, dict):
            return super().__eq__(Attributes(other))
        return super().__eq__(other)


class Case:
    """
    A case is a collection of attributes that represents an instance that can be classified by inferring new attributes
    or additional attribute values for the case.
    """

    def __init__(self, id_: str, attributes: List[Attribute],
                 conclusions: Optional[List[Attribute]] = None,
                 targets: Optional[List[Attribute]] = None):
        """
        Create a case.

        :param id_: The id of the case.
        :param attributes: The attributes of the case.
        :param conclusions: The conclusions that has been made about the case.
        """
        self.attributes = Attributes({a.name: a for a in attributes})
        self.id_ = id_
        self.conclusions: Optional[List[Attribute]] = conclusions
        self.targets: Optional[List[Attribute]] = targets

    def remove_attribute(self, attribute_name: str):
        if attribute_name in self:
            del self.attributes[attribute_name]

    def add_attributes(self, attributes: List[Attribute]):
        if not attributes:
            return
        attributes = attributes if isinstance(attributes, list) else [attributes]
        for attribute in attributes:
            self.add_attribute(attribute)

    def add_attribute(self, attribute: Attribute):
        self.attributes[attribute.name] = attribute

    @property
    def attribute_values(self):
        return [a.value for a in self.attributes.values()]

    @property
    def attributes_list(self):
        return list(self.attributes.values())

    def __eq__(self, other):
        return self.attributes == other.attributes

    def __getitem__(self, attribute_or_attribute_name: Union[str, Attribute]) -> Attribute:
        return self.attributes.get(attribute_or_attribute_name, None)

    def __sub__(self, other):
        return {k: self.attributes[k] for k in self.attributes
                if self.attributes[k] != other.attributes[k]}

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.attributes
        elif isinstance(item, type) and issubclass(item, Attribute):
            return item.__name__ in self.attributes
        elif isinstance(item, Attribute):
            return item.name in self.attributes and self.attributes[item.name] == item

    @staticmethod
    def ljust(s, sz=15):
        return str(s).ljust(sz)

    def print_all_names(self, all_names: List[str], max_len: int,
                        target_types: Optional[List[Type[Attribute]]] = None,
                        conclusion_types: Optional[List[Type[Attribute]]] = None) -> int:
        """
        Print all attribute names.

        :param all_names: list of names.
        :param max_len: maximum length.
        :param target_types: list of target types.
        :param conclusion_types: list of category types.
        :return: maximum length.
        """
        all_names_str, max_len = self.get_all_names_str(all_names, max_len, target_types, conclusion_types)
        print(all_names_str)
        return max_len

    def print_values(self, all_names: Optional[List[str]] = None,
                     targets: Optional[List[Attribute]] = None,
                     is_corner_case: bool = False,
                     ljust_sz: int = 15,
                     conclusions: Optional[List[Attribute]] = None):
        print(self.get_values_str(all_names, targets, is_corner_case, conclusions, ljust_sz))

    def __str__(self):
        names, ljust = self.get_all_names_and_max_len()
        row1, ljust = self.get_all_names_str(names, ljust)
        row2 = self.get_values_str(names, ljust_sz=ljust)
        return "\n".join([row1, row2])

    def get_all_names_str(self, all_names: List[str], max_len: int,
                          target_types: Optional[List[Type[Attribute]]] = None,
                          conclusion_types: Optional[List[Type[Attribute]]] = None) -> Tuple[str, int]:
        """
        Get all attribute names, target names and conclusion names.

        :param all_names: list of names.
        :param max_len: maximum length.
        :param target_types: list of target types.
        :param conclusion_types: list of category types.
        :return: string of names, maximum length.
        """
        if conclusion_types or self.conclusions:
            conclusion_types = conclusion_types or list(map(type, self.conclusions))
        category_names = []
        if conclusion_types:
            category_types = conclusion_types or [Attribute]
            category_names = [category_type.__name__.lower() for category_type in category_types]

        if target_types or self.targets:
            target_types = target_types if target_types else list(map(type, self.targets))
        target_names = []
        if target_types:
            target_names = [f"target_{target_type.__name__.lower()}" for target_type in target_types]

        curr_max_len = max(max_len, max([len(name) for name in all_names + category_names + target_names])+2)
        names_row = self.ljust(f"names: ", sz=curr_max_len)
        names_row += self.ljust("ID", sz=curr_max_len)
        names_row += "".join([f"{self.ljust(name, sz=curr_max_len)}" for name in all_names + category_names + target_names])
        return names_row, curr_max_len

    def get_all_names_and_max_len(self, all_attributes: Optional[List[Attribute]] = None) -> Tuple[List[str], int]:
        """
        Get all attribute names and the maximum length of the names and values.

        :param all_attributes: list of attributes
        :return: list of names and the maximum length
        """
        all_attributes = all_attributes if all_attributes else self.attributes_list
        all_names = list(OrderedSet([a.name for a in all_attributes]))
        max_len = max([len(name) for name in all_names])
        max_len = max(max_len, max([len(str(a.value)) for a in all_attributes])) + 4
        return all_names, max_len

    def get_values_str(self, all_names: Optional[List[str]] = None,
                       targets: Optional[List[Attribute]] = None,
                       is_corner_case: bool = False,
                       conclusions: Optional[List[Attribute]] = None,
                       ljust_sz: int = 15) -> str:
        """
        Get the string representation of the values of the case.
        """
        all_names = list(self.attributes.keys()) if not all_names else all_names
        targets = targets if targets else self.targets
        if targets:
            targets = targets if isinstance(targets, list) else [targets]
        case_row = self.get_id_and_attribute_values_str(all_names, is_corner_case, ljust_sz)
        case_row += self.get_conclusions_str(conclusions, ljust_sz)
        case_row += self.get_targets_str(targets, ljust_sz)
        return case_row

    def get_id_and_attribute_values_str(self, all_names: Optional[List[str]] = None,
                                        is_corner_case: bool = False,
                                        ljust_sz: int = 15) -> str:
        """
        Get the string representation of the id and names of the case.

        :param all_names: The names of the attributes to include in the string.
        :param is_corner_case: Whether the case is a corner case.
        :param ljust_sz: The size of the ljust.
        """
        all_names = list(self.attributes.keys()) if not all_names else all_names
        if is_corner_case:
            case_row = self.ljust(f"corner case: ", sz=ljust_sz)
        else:
            case_row = self.ljust(f"case: ", sz=ljust_sz)
        case_row += self.ljust(self.id_, sz=ljust_sz)
        case_row += "".join([f"{self.ljust(self[name].value if name in self.attributes else '', sz=ljust_sz)}"
                             for name in all_names])
        return case_row

    def get_targets_str(self, targets: Optional[List[Attribute]] = None, ljust_sz: int = 15) -> str:
        """
        Get the string representation of the targets of the case.
        """
        targets = targets if targets else self.targets
        return self._get_categories_str(targets, ljust_sz)

    def get_conclusions_str(self, conclusions: Optional[List[Attribute]] = None, ljust_sz: int = 15) -> str:
        """
        Get the string representation of the conclusions of the case.
        """
        conclusions = conclusions if conclusions else self.conclusions
        return self._get_categories_str(conclusions, ljust_sz)

    def _get_categories_str(self, categories: List[Attribute], ljust_sz: int = 15) -> str:
        """
        Get the string representation of the categories of the case.
        """
        if not categories:
            return ""
        categories_str = [self.ljust(c.value, sz=ljust_sz) for c in categories]
        return "".join(categories_str) if len(categories_str) > 1 else categories_str[0]

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        conclusions_cp = self.conclusions.copy() if self.conclusions else None
        return Case(self.id_, self.attributes_list.copy(), conclusions_cp)
