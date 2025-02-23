from __future__ import annotations

from abc import abstractmethod, ABC
from collections import UserDict
from dataclasses import dataclass
from enum import Enum, auto

from ordered_set import OrderedSet
from typing_extensions import Any, Tuple, Optional, List, Dict, Type, Union, Sequence, Set, Self, get_origin, get_args, \
    get_type_hints

from .failures import InvalidOperator
from .utils import make_set, make_value_or_raise_error


class RDRMode(Enum):
    Propositional = auto()
    """
    Propositional mode, the mode where the rules are propositional.
    """
    Relational = auto()
    """
    Relational mode, the mode where the rules are relational.
    """


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
    Iterable = auto()
    """
    Iterable value type (eg. [1, 2, 3]).
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
    _range: Union[set, Range] = None
    """
    The range of the attribute, this can be a set of possible values or a range of numeric values (int, float).
    """

    _registry: Dict[str, Type[Self]] = {}
    """
    A dictionary of all dynamically created subclasses of the attribute class.
    """

    @classmethod
    def create_attribute(cls, name: str, mutually_exclusive: bool, value_type: CategoryValueType,
                         range_: Union[set, Range], **kwargs) \
            -> Type[Self]:
        """
        Create a new attribute subclass.

        :param name: The name of the attribute.
        :param mutually_exclusive: Whether the attribute is mutually exclusive.
        :param value_type: The type of the value of the attribute.
        :param range_: The range of the attribute.
        :return: The new attribute subclass.
        """
        kwargs.update(mutually_exclusive=mutually_exclusive, value_type=value_type, _range=range_)
        if name in cls._registry:
            if not cls._registry[name].mutually_exclusive == mutually_exclusive:
                print(f"Mutually exclusive of {name} is different from {cls._registry[name].mutually_exclusive}.")
                cls._registry[name].mutually_exclusive = mutually_exclusive
            if not cls._registry[name].value_type == value_type:
                raise ValueError(f"Value type of {name} is different from {cls._registry[name].value_type}.")
            if not cls._registry[name].within_range(range_):
                if isinstance(cls._registry[name]._range, set):
                    cls._registry[name]._range.union(range_)
                else:
                    raise ValueError(f"Range of {name} is different from {cls._registry[name]._range}.")
            return cls._registry[name]
        new_attribute_type: Type[Self] = type(name.lower(), (cls,), {}, **kwargs)
        cls.register(new_attribute_type)
        return new_attribute_type

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
        super().__init_subclass__()
        cls.name = cls.__name__.lower()

        mutually_exclusive = kwargs.get("mutually_exclusive", None)
        value_type = kwargs.get("value_type", None)
        _range = kwargs.get("_range", None)

        cls.mutually_exclusive = mutually_exclusive if mutually_exclusive else cls.mutually_exclusive
        cls.value_type = value_type if value_type else cls.value_type
        cls._range = _range if _range else cls._range

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
        value = self.make_value(value)
        if not self.mutually_exclusive:
            self._value = make_set(value)
        else:
            self._value = value

    @classmethod
    def make_value(cls, value: Any) -> Any:
        """
        Make a value for the attribute.

        :param value: The value to make.
        """
        if not cls.is_possible_value(value):
            raise ValueError(f"Value {value} is not a possible value for {cls.__name__} with range {cls._range}.")
        if cls.value_type == CategoryValueType.Iterable:
            if not hasattr(value, "__iter__") or isinstance(value, str):
                value = [value]
        else:
            value = make_value_or_raise_error(value)
        return cls._make_value(value)

    @classmethod
    @abstractmethod
    def _make_value(cls, value: Any) -> Any:
        """
        Make a value for the attribute.

        :param value: The value to make.
        """
        pass

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
    @abstractmethod
    def is_possible_value(cls, value: Any) -> bool:
        """
        Check if a value is a possible value for the attribute or if it can be converted to a possible value.

        :param value: The value to check.
        """
        pass

    @classmethod
    def is_within_range(cls, value: Union[set, Range, Any]) \
            -> bool:
        """
        Check if a value is within the range of the attribute.

        :param value: The value to check.
        :return: Boolean indicating whether the value is within the range or not.
        """
        if isinstance(cls._range, set):
            if hasattr(value, "__iter__") and not isinstance(value, str):
                return set(value).issubset(cls._range)
            elif isinstance(value, Range):
                return False
            elif isinstance(value, str):
                return value.lower() in cls._range
            else:
                return value in cls._range
        else:
            return value in cls._range

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

    @classmethod
    def _make_value(cls, value: Any) -> int:
        return int(value)

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.isdigit() and cls.is_within_range(int(value))
        else:
            return isinstance(value, int) and cls.is_within_range(value)


class Continuous(Attribute):
    """
    A continuous attribute is an attribute that has a value that is a continuous category.
    """
    mutually_exclusive: bool = False
    value_type = CategoryValueType.Continuous
    _range: Range = Range(-float("inf"), float("inf"), min_closed=False, max_closed=False)

    @classmethod
    def _make_value(cls, value: Any) -> float:
        return float(value)

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.replace(".", "", 1).isdigit() and cls.is_within_range(float(value))
        return isinstance(value, (float, int)) and cls.is_within_range(value)


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
        return list(cls._value2member_map_.keys())

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
    _range: Set[Union[str, type]] = None
    Values: Type[CategoricalValue]

    def __init_subclass__(cls, **kwargs):
        """
        Create the Values enum class for the categorical attribute, this enum class contains all the possible values
        of the attribute.
        Note: This method is called when a subclass of Categorical is created (not when an instance is created).
        """
        super().__init_subclass__(**kwargs)
        if not cls._range:
            cls._range = set()
        cls.create_values()

    def __init__(self, value: Union[Categorical.Values, str]):
        super().__init__(value)

    @classmethod
    def _make_value(cls, value: Union[str, Categorical.Values]) -> Categorical.Values:
        if len(cls._range) == 0:
            if isinstance(value, str):
                cls.add_new_category(value)
            else:
                cls.add_new_category(type(value))
            return cls._make_value(value)
        if isinstance(value, str) and len(cls._range) > 0 and type(list(cls._range)[0]) == str:
            return cls.Values[value.lower()]
        elif isinstance(value, cls.Values) or any(isinstance(v, type) and isinstance(value, v) for v in cls.Values.to_list()):
            return value
        else:
            raise ValueError(f"Value {value} should be a string or a CategoricalValue.")

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if len(cls._range) == 0:
            if isinstance(value, str):
                cls.add_new_category(value)
            else:
                cls.add_new_category(type(value))
            return cls.is_possible_value(value)
        if isinstance(value, str) and len(cls._range) > 0 and type(list(cls._range)[0]) == str:
            return cls.is_within_range(value.lower())
        elif isinstance(value, cls.Values) or any(isinstance(value, v) for v in cls.Values.to_list()):
            return True
        else:
            return False

    @classmethod
    def from_str(cls, category: str):
        return cls(cls.Values[category.lower()])

    @classmethod
    def from_strs(cls, categories: List[str]):
        return [cls.from_str(c) for c in categories]

    @classmethod
    def add_new_categories(cls, categories: List[str]):
        for category in categories:
            cls.add_new_category(category)

    @classmethod
    def add_new_category(cls, category: str):
        if isinstance(category, str):
            cls._range.add(category.lower())
        elif isinstance(category, type):
            cls._range.add(category)
        else:
            raise ValueError(f"Category {category} should be a string or a type.")
        cls.create_values()

    @classmethod
    def create_values(cls):
        if all(isinstance(c, str) for c in cls._range):
            cls.Values = CategoricalValue(f"{cls.__name__}Values", {c.lower(): c.lower() for c in cls._range})
        else:
            cls.Values = CategoricalValue(f"{cls.__name__}Values",
                                          {c.__name__.lower(): c for c in cls._range})


class ListOf(Attribute, ABC):
    """
    A list of attribute is an attribute that has a value that is a list of other attributes, but all the attributes in
    the list must have the same type.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Iterable
    _range: Set[Attribute]
    element_type: Type[Attribute]
    list_size: Optional[int] = None

    def __init__(self, value: List[element_type]):
        super().__init__(value)

    @classmethod
    def create_attribute(cls, name: str, element_type: Type[Attribute],
                         list_size: Optional[int] = None) -> Type[Self]:
        """
        Create a new attribute subclass that is a list of other attributes ot type _element_type.

        :param name: The name of the attribute.
        :param element_type: The type of the elements in the list.
        :param list_size: The size of the list.
        :return: The new attribute subclass.
        """
        return super().create_attribute(name, cls.mutually_exclusive, cls.value_type, make_set(element_type),
                                        element_type=element_type, list_size=list_size)

    def __init_subclass__(cls, **kwargs):
        """
        """
        super().__init_subclass__(**kwargs)
        element_type = kwargs.get("element_type", None)
        list_size = kwargs.get("list_size", None)
        if not element_type:
            raise ValueError("ListOf subclasses must have an element_type.")
        cls.element_type = element_type
        cls.list_size = list_size

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if not hasattr(value, "__iter__") or isinstance(value, str):
            value = [value]
        if cls.list_size and len(value) != cls.list_size:
            return False
        else:
            return all(cls.element_type.is_possible_value(v) for v in value)

    @classmethod
    def _make_value(cls, value: Any) -> List[element_type]:
        if cls.list_size and len(value) != cls.list_size:
            raise ValueError(f"Value {value} should be a list with size {cls.list_size},"
                             f"got list of size {len(value)} instead.")
        value = [cls.element_type.make_value(v) if not isinstance(v, cls.element_type) else v for v in value]
        return value


class DictOf(Attribute, ABC):
    """
    A dictionary of attribute is an attribute that has a value that is a dictionary of other attributes, but all the
    attributes in the dictionary must have the same type.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Iterable
    _range: Set[Attribute]
    element_type: Type[Attribute]

    def __init__(self, value: Dict[str, element_type]):
        super().__init__(value)

    @classmethod
    def create_attribute(cls, name: str, element_type: Type[Attribute]) -> Type[Self]:
        """
        Create a new attribute subclass that is a dictionary of other attributes ot type _element_type.

        :param name: The name of the attribute.
        :param element_type: The type of the elements in the dictionary.
        :return: The new attribute subclass.
        """
        return super().create_attribute(name, cls.mutually_exclusive, cls.value_type, make_set(element_type),
                                        element_type=element_type)

    def __init_subclass__(cls, **kwargs):
        """
        """
        super().__init_subclass__(**kwargs)
        element_type = kwargs.get("element_type", None)
        if not element_type:
            raise ValueError("DictOf subclasses must have an element_type.")
        cls.element_type = element_type

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        else:
            return all(cls.element_type.is_possible_value(v) for v in value.values())

    @classmethod
    def _make_value(cls, value: Any) -> Dict[str, element_type]:
        value = {k: cls.element_type.make_value(v) if not isinstance(v, cls.element_type) else v for k, v in value.items()}
        return value


class Bool(Attribute):
    """
    A binary attribute is an attribute that has a value that is a binary category.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Binary
    _range: set = {True, False}

    def __init__(self, value: Union[bool, str, int, float]):
        super().__init__(value)

    @classmethod
    def _make_value(cls, value: Any) -> bool:
        return bool(value)

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.lower() in ["true", "false"]
        if isinstance(value, bool):
            return True
        if isinstance(value, int):
            return value in [0, 1]
        if isinstance(value, float):
            return value in [0.0, 1.0]
        return False


class Unary(Attribute):
    """
    A unary attribute is an attribute that has a value that is a unary category.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Unary
    _range: set

    def __init__(self):
        super().__init__(self.__class__.__name__)

    @classmethod
    def _make_value(cls, value: Any) -> str:
        return cls.__name__

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.lower() == cls.__name__.lower()
        return False


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

    @classmethod
    def from_object(cls, obj: Any, attributes: Optional[List[Attribute]] = None,
                    conclusions: Optional[List[Attribute]] = None,
                    targets: Optional[List[Attribute]] = None) -> Case:
        """
        Create a case from an object.

        :param obj: The object to create the case from.
        :param attributes: The attributes of the case.
        :param conclusions: The conclusions that has been made about the case.
        :param targets: The targets of the case.
        :return: The case.
        """
        if not attributes:
            attributes = cls.get_attributes_from_object(obj)
        return cls(obj.__class__.__name__, attributes, conclusions, targets)

    @staticmethod
    def get_attributes_from_object(obj: Any) -> List[Attribute]:
        """
        Get the attributes of an object.

        :param obj: The object to get the attributes from.
        :return: The attributes of the object.
        """
        attributes = []
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name)
            if not attr_name.startswith("_") and not callable(attr):
                matched_attribute = Case.get_or_create_matching_attribute(attr, attr_name)
                attributes.append(matched_attribute)
        return attributes

    @staticmethod
    def get_or_create_matching_attribute(attr_value: Any, attr_name: str) -> Attribute:
        """
        Get or create a matching attribute type for an attribute value.

        :param attr_value: The value of the attribute.
        :param attr_name: The name of the attribute.
        :return: The matching attribute type instantiated with the attribute value.
        """
        iterable = hasattr(attr_value, "__iter__") and not isinstance(attr_value, str)
        if iterable:
            element_attr_name = f"{attr_name}_element"
            iterable_type = ListOf
            values = attr_value
            if type(attr_value) == dict:
                values = list(attr_value.values())
                iterable_type = DictOf
            if len(values) == 0:
                _range = make_set(type(values[0])) if len(values) > 0 else make_set(values)
                attr_type = Categorical.create_attribute(element_attr_name, False, CategoryValueType.Nominal,
                                                         _range)
                print(f"Created categorical attribute {element_attr_name} from iterable {attr_value}"
                      f" as there are no values.")
            elif all(Integer.is_possible_value(v) for v in values):
                attr_type = Integer
            elif all(Continuous.is_possible_value(v) for v in values):
                attr_type = Continuous
            elif all(Bool.is_possible_value(v) for v in values):
                attr_type = Bool
            else:
                if list(set(map(type, values))) == [type(values[0])]:
                    attr_type = type(Case.get_or_create_matching_attribute(values[0], element_attr_name))
                else:
                    raise ValueError(f"Cannot create attribute: Values in iterable {attr_name} are of different types, "
                                     f"{list(set(map(type, values)))}.")
            attr_type = iterable_type.create_attribute(attr_name, attr_type)
        else:
            if Integer.is_possible_value(attr_value):
                attr_type = Integer
            elif Continuous.is_possible_value(attr_value):
                attr_type = Continuous
            elif Bool.is_possible_value(attr_value):
                attr_type = Bool
            else:
                attr_type = Categorical.create_attribute(attr_name, False, CategoryValueType.Nominal,
                                                         make_set(type(attr_value)))
        return attr_type(attr_value)

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
        self[attribute.name] = attribute

    def __setitem__(self, attribute_name: str, attribute: Attribute):
        self.attributes[attribute_name] = attribute

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

        curr_max_len = max(max_len, max([len(name) for name in all_names + category_names + target_names]) + 2)
        names_row = self.ljust(f"names: ", sz=curr_max_len)
        names_row += self.ljust("ID", sz=curr_max_len)
        names_row += "".join(
            [f"{self.ljust(name, sz=curr_max_len)}" for name in all_names + category_names + target_names])
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
