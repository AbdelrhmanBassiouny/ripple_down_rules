from __future__ import annotations

from abc import abstractmethod, ABC
from enum import Enum, auto

from orderedset import OrderedSet
from typing_extensions import Any, Callable, Tuple, Optional, List, Dict, Type

from .failures import InvalidOperator


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


# class Category:
#     """
#     A category is an abstract concept that represents a class or a label. In a classification problem, a category
#     represents a class or a label that a case can be classified into. In RDR it is referred to as a conclusion.
#     It is important to know that a concept can be an attribute or a category depending on the context, for example,
#     in the case when one is trying to infer the species of an animal, the species is a category, but when one is trying
#     to infer if a species flies or not, the concept of flying becomes a category while the species becomes an attribute.
#     """
#     mutually_exclusive: bool = False
#     value_type: CategoryValueType = CategoryValueType.Nominal
#
#     def __init__(self, value: Any):
#         self.value = value
#
#     @property
#     def value(self):
#         return self._value
#
#     @value.setter
#     def value(self, value: Any):
#         if not self.mutually_exclusive and not isinstance(value, set):
#             value = {value}
#         self._value = value
#
#     def __eq__(self, other):
#         if not isinstance(other, Category):
#             return False
#         if isinstance(self.value, set) and not isinstance(other.value, set):
#             return other.value in self.value
#         elif not isinstance(self.value, set) and isinstance(other.value, set):
#             return self.value in other.value and len(other.value) == 1
#         return self.__class__ == other.__class__ and self.value == other.value
#
#     def __hash__(self):
#         return hash(self.value) if not isinstance(self.value, set) else hash(frozenset(self.value))
#
#     def __str__(self):
#         return f"{type(self).__name__}({self.value})"
#
#     def __repr__(self):
#         return self.__str__()
#
#
# class Stop(Category):
#     """
#     A stop category is a special category that represents the stopping of the classification to prevent a wrong
#     conclusion from being made.
#     """
#     mutually_exclusive = True
#
#     def __init__(self):
#         super().__init__("null")
#
#
# class Species(Category):
#     """
#     A species category is a category that represents the species of an animal.
#     """
#     mutually_exclusive: bool = True
#
#
# class Habitat(Category):
#     """
#     A habitat category is a category that represents the habitat of an animal.
#     """
#     ...


class Attribute:
    """
    An attribute is a name-value pair that represents a feature of a case.
    an attribute can be used to compare two cases and to make a conclusion about a case.
    """

    def __init__(self, name: str, value: Any, mutually_exclusive: bool = False,
                 value_type: CategoryValueType = CategoryValueType.Nominal):
        """
        Create an attribute.

        :param name: The name of the attribute.
        :param value: The value of the attribute.
        :param mutually_exclusive: Whether the value of the attribute is mutually exclusive.
        :param value_type: The type of the value of the attribute.
        """
        self.name = name.lower()
        self.mutually_exclusive: bool = mutually_exclusive
        self.value_type: CategoryValueType = value_type
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: Any):
        if not self.mutually_exclusive and not isinstance(value, set):
            value = {value}
        self._value = value

    def __eq__(self, other: Attribute):
        if not isinstance(other, Attribute):
            return False
        if self.name != other.name:
            return False
        if isinstance(self.value, set) and not isinstance(other.value, set):
            return other.value in self.value
        elif isinstance(self.value, set) and isinstance(other.value, set):
            return other.value.issubset(self.value)
        else:
            return self.value == other.value

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f"{self.name}: {self.value}"

    def __repr__(self):
        return self.__str__()


class Stop(Attribute):
    """
    A stop category is a special category that represents the stopping of the classification to prevent a wrong
    conclusion from being made.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__, "null", mutually_exclusive=True,
                         value_type=CategoryValueType.Unary)
        

class SpeciesValue(Enum):
    Mammal = auto()
    Bird = auto()
    Reptile = auto()
    Fish = auto()
    Amphibian = auto()
    Insect = auto()
    Molusc = auto()

    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, value: str):
        return cls[value.capitalize()]

    @classmethod
    def from_strs(cls, values: List[str]):
        return [cls.from_str(value) for value in values]


class Species(Attribute):
    """
    A species category is a category that represents the species of an animal.
    """
    def __init__(self, species: SpeciesValue):
        super().__init__(self.__class__.__name__, species, mutually_exclusive=True,
                         value_type=CategoryValueType.Nominal)


class HabitatValue(Enum):
    Land = auto()
    Water = auto()
    Air = auto()


class Habitat(Attribute):
    """
    A habitat category is a category that represents the habitat of an animal.
    """
    def __init__(self, habitat: HabitatValue):
        super().__init__(self.__class__.__name__, habitat, mutually_exclusive=True,
                         value_type=CategoryValueType.Nominal)


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


def str_to_operator_fn(rule_str: str) -> Tuple[Optional[str], Optional[str], Optional[Callable]]:
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
    A condition is a constraint on an attribute that must be satisfied for a case to be classified into a category.
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


class Attributes(dict):
    """
    A collection of attributes that represents a set of constraints on a case.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k.lower()] = v

    def __getitem__(self, item):
        return super().__getitem__(item.lower())

    def __setitem__(self, key, value):
        super().__setitem__(key.lower(), value)

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
    A case is a collection of attributes that represents an instance that can be classified into a category or
    multiple categories.
    """

    def __init__(self, id_: str, attributes: List[Attribute],
                 conclusions: Optional[List[Category]] = None,
                 targets: Optional[List[Category]] = None):
        """
        Create a case.

        :param id_: The id of the case.
        :param attributes: The attributes of the case.
        :param conclusions: The conclusions that has been made about the case.
        """
        self.attributes = Attributes({a.name: a for a in attributes})
        self.id_ = id_
        self.conclusions: Optional[List[Category]] = conclusions
        self.targets: Optional[List[Category]] = targets

    def remove_attribute_equivalent_to_category(self, category: Category):
        if category in self:
            self.remove_attribute_equivalent_to_category_type(type(category))

    def remove_attribute_equivalent_to_category_type(self, category_type: Type[Category]):
        if category_type in self:
            self.remove_attribute(category_type.__name__)

    def remove_attribute(self, attribute_name: str):
        if attribute_name in self:
            del self.attributes[attribute_name.lower()]

    def add_attributes_from_categories(self, categories: List[Category]):
        if not categories:
            return
        categories = categories if isinstance(categories, list) else [categories]
        for category in categories:
            self.add_attribute_from_category(category)

    def add_attribute_from_category(self, category: Category):
        self.add_attribute(Attribute.from_category(category))

    def add_attribute(self, attribute: Attribute):
        if attribute.name in self.attributes:
            if isinstance(attribute.value, set):
                self.attributes[attribute.name].value.update(attribute.value)
            else:
                raise ValueError(f"Attribute {attribute.name} already exists in the case.")
        else:
            self.attributes[attribute.name] = attribute

    @property
    def attribute_values(self):
        return [a.value for a in self.attributes.values()]

    @property
    def attributes_list(self):
        return list(self.attributes.values())

    def __eq__(self, other):
        return self.attributes == other.attributes

    def __getitem__(self, attribute_name):
        return self.attributes.get(attribute_name.lower(), None)

    def __sub__(self, other):
        return {k: self.attributes[k] for k in self.attributes
                if self.attributes[k] != other.attributes[k]}

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.attributes
        elif isinstance(item, Category):
            return Attribute.from_category(item) in self
        elif isinstance(item, type) and issubclass(item, Category):
            return item.__name__ in self.attributes
        elif isinstance(item, Attribute):
            return item.name in self.attributes and self.attributes[item.name] == item

    @staticmethod
    def ljust(s, sz=15):
        return str(s).ljust(sz)

    def print_all_names(self, all_names: List[str], max_len: int,
                        target_types: Optional[List[Type[Category]]] = None,
                        conclusion_types: Optional[List[Type[Category]]] = None):
        """
        Print all attribute names.

        :param all_names: list of names.
        :param max_len: maximum length.
        :param target_types: list of target types.
        :param conclusion_types: list of category types.
        """
        print(self.get_all_names_str(all_names, max_len, target_types, conclusion_types))

    def print_values(self, all_names: Optional[List[str]] = None,
                     targets: Optional[List[Category]] = None,
                     is_corner_case: bool = False,
                     ljust_sz: int = 15,
                     conclusions: Optional[List[Category]] = None):
        print(self.get_values_str(all_names, targets, is_corner_case, conclusions, ljust_sz))

    def __str__(self):
        names, ljust = self.get_all_names_and_max_len()
        row1 = self.get_all_names_str(names, ljust)
        row2 = self.get_values_str(names, ljust_sz=ljust)
        return "\n".join([row1, row2])

    def get_all_names_str(self, all_names: List[str], max_len: int,
                            target_types: Optional[List[Type[Category]]] = None,
                            conclusion_types: Optional[List[Type[Category]]] = None) -> str:
        """
        Get all attribute names, target names and conclusion names.

        :param all_names: list of names.
        :param max_len: maximum length.
        :param target_types: list of target types.
        :param conclusion_types: list of category types.
        :return: string of names.
        """
        if conclusion_types or self.conclusions:
            conclusion_types = conclusion_types or list(map(type, self.conclusions))
        category_names = []
        if conclusion_types:
            category_types = conclusion_types or [Category]
            category_names = [category_type.__name__.lower() for category_type in category_types]

        if target_types or self.targets:
            target_types = target_types if target_types else list(map(type, self.targets))
        target_names = []
        if target_types:
            target_names = [f"target_{target_type.__name__.lower()}" for target_type in target_types]

        names_row = self.ljust(f"names: ", sz=max_len)
        names_row += self.ljust("ID", sz=max_len)
        names_row += "".join([f"{self.ljust(name, sz=max_len)}" for name in all_names + category_names + target_names])
        return names_row

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
                          targets: Optional[List[Category]] = None,
                            is_corner_case: bool = False,
                            conclusions: Optional[List[Category]] = None,
                       ljust_sz: int = 15) -> str:
        """
        Get the string representation of the values of the case.
        """
        all_names = list(self.attributes.keys()) if not all_names else all_names
        targets = targets if targets else self.targets
        if targets:
            targets = targets if isinstance(targets, list) else [targets]
        case_row = self.get_id_and_attribute_values_str(all_names, is_corner_case, ljust_sz)
        case_row += self.get_targets_str(targets, ljust_sz)
        case_row += self.get_conclusions_str(conclusions, ljust_sz)
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

    def get_targets_str(self, targets: Optional[List[Category]] = None, ljust_sz: int = 15) -> str:
        """
        Get the string representation of the targets of the case.
        """
        targets = targets if targets else self.targets
        return self._get_categories_str(targets, ljust_sz)

    def get_conclusions_str(self, conclusions: Optional[List[Category]] = None, ljust_sz: int = 15) -> str:
        """
        Get the string representation of the conclusions of the case.
        """
        conclusions = conclusions if conclusions else self.conclusions
        return self._get_categories_str(conclusions, ljust_sz)

    def _get_categories_str(self, categories: List[Category], ljust_sz: int = 15) -> str:
        """
        Get the string representation of the categories of the case.
        """
        if not categories:
            return ""
        categories_str = [self.ljust(c.value, sz=ljust_sz) for c in categories]
        return ",".join(categories_str) if len(categories_str) > 1 else categories_str[0]

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        conclusions_cp = self.conclusions.copy() if self.conclusions else None
        return Case(self.id_, self.attributes_list.copy(), conclusions_cp)
