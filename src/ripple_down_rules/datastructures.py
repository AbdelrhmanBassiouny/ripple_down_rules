from __future__ import annotations

from abc import abstractmethod, ABC
from enum import Enum, auto

from typing_extensions import Any, Callable, Tuple, Optional, List, Dict

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


class Category:
    """
    A category is an abstract concept that represents a class or a label. In a classification problem, a category
    represents a class or a label that a case can be classified into. In RDR it is referred to as a conclusion.
    It is important to know that a concept can be an attribute or a category depending on the context, for example,
    in the case when one is trying to infer the species of an animal, the species is a category, but when one is trying
    to infer if a species flies or not, the concept of flying becomes a category while the species becomes an attribute.
    """

    def __init__(self, value: Any):
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, Category):
            return False
        return type(self) == type(other) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return f"{type(self).__name__}({self.value})"

    def __repr__(self):
        return self.__str__()


class Stop(Category):
    """
    A stop category is a special category that represents the stopping of the classification to prevent a wrong
    conclusion from being made.
    """

    def __init__(self):
        super().__init__("null")


class Attribute:
    """
    An attribute is a name-value pair that represents a feature of a case.
    an attribute can be used to compare two cases and to make a conclusion about a case.
    """

    def __init__(self, name: str, value: Any):
        self.name = name.lower()
        self.value = value

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def __hash__(self):
        return hash((self.name, self.value))

    def __str__(self):
        return f"{self.name}: {self.value}"

    def __repr__(self):
        return self.__str__()


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
    operators = [LessEqual(), GreaterEqual(), Equal(), Less(), Greater()]
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


class Case:
    """
    A case is a collection of attributes that represents an instance that can be classified into a category or
    multiple categories.
    """
    def __init__(self, id_: str, attributes: List[Attribute],
                 conclusions: Optional[List[Category]] = None):
        """
        Create a case.

        :param id_: The id of the case.
        :param attributes: The attributes of the case.
        :param conclusions: The conclusions that has been made about the case.
        """
        self.attributes = {a.name: a for a in attributes}
        self.id_ = id_
        self.conclusions: Optional[List[Category]] = conclusions

    def add_attributes_from_categories(self, categories: List[Category]):
        for category in categories:
            self.add_attribute_from_category(category)

    def add_attribute_from_category(self, category: Category):
        self.add_attribute(Attribute(type(category).__name__, category.value))

    def add_attribute(self, attribute: Attribute):
        if attribute.name in self.attributes:
            raise ValueError(f"Attribute {attribute.name} already exists in the case.")
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
        return self.attributes.get(attribute_name, None)

    def __sub__(self, other):
        return {k: self.attributes[k] for k in self.attributes
                if self.attributes[k] != other.attributes[k]}

    def __contains__(self, item):
        if isinstance(item, str):
            return item.lower() in self.attributes
        elif isinstance(item, Category):
            return type(item).__name__.lower() in self.attributes
        elif issubclass(item, Category):
            return item.__name__.lower() in self.attributes
        elif isinstance(item, Attribute):
            return item.name.lower() in self.attributes and self.attributes[item.name.lower()] == item

    @staticmethod
    def ljust(s, sz=15):
        return str(s).ljust(sz)

    def print_values(self, all_names: Optional[List[str]] = None,
                     target: Optional[Category] = None,
                     is_corner_case: bool = False,
                     ljust_sz: int = 15,
                     conclusions: Optional[List[Category]] = None):
        all_names = list(self.attributes.keys()) if not all_names else all_names
        if is_corner_case:
            case_row = self.ljust(f"corner case: ", sz=ljust_sz)
        else:
            case_row = self.ljust(f"case: ", sz=ljust_sz)
        case_row += self.ljust(self.id_, sz=ljust_sz)
        case_row += "".join([f"{self.ljust(self[name].value, sz=ljust_sz)}"
                             for name in all_names])
        if target:
            case_row += f"{self.ljust(target.value, sz=ljust_sz)}"
        if conclusions:
            case_row += ",".join([f"{self.ljust(c.value, sz=ljust_sz)}" for c in conclusions])
        print(case_row)

    def __str__(self):
        names = list(self.attributes.keys())
        ljust = max([len(name) for name in names])
        ljust = max(ljust, max([len(str(a.value)) for a in self.attributes.values()])) + 2
        row1 = f"Case {self.id_} with attributes: \n"
        row2 = [f"{name.ljust(ljust)}" for name in names]
        row2 = "".join(row2)
        if self.conclusions:
            row2 += "Conclusions".ljust(ljust)
        row2 += "\n"
        row3 = [f"{str(self.attributes[name].value).ljust(ljust)}" for name in names]
        row3 = "".join(row3)
        if self.conclusions:
            row3 += ",".join([c.value for c in self.conclusions])
        return row1 + row2 + row3 + "\n"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        conclusions_cp = self.conclusions.copy() if self.conclusions else None
        return Case(self.id_, self.attributes_list.copy(), conclusions_cp)
