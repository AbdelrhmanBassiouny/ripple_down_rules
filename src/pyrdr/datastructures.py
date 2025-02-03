from enum import Enum, auto

from typing_extensions import Any, Callable, Tuple, Optional, List

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


class Category:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Stop(Category):
    def __init__(self):
        super().__init__("null")


class Attribute:
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def __hash__(self):
        return hash((self.name, self.value))


class Operator:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func

    def __call__(self, x: Any, y: Any) -> bool:
        return self.func(x, y)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class Equal(Operator):
    def __init__(self):
        super().__init__("==", lambda x, y: x == y)


class Greater(Operator):
    def __init__(self):
        super().__init__(">", lambda x, y: x > y)


class GreaterEqual(Operator):
    def __init__(self):
        super().__init__(">=", lambda x, y: x >= y)


class Less(Operator):
    def __init__(self):
        super().__init__("<", lambda x, y: x < y)


class LessEqual(Operator):

    def __init__(self):
        super().__init__("<=", lambda x, y: x <= y)


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
    def __init__(self, name: str, value: Any, operator: Operator):
        self.name = name
        self.value = value
        self.operator = operator

    def __call__(self, x: Any) -> bool:
        return self.operator(x, self.value)

    def __str__(self):
        return f"{self.name} {self.operator} {self.value}"

    def __repr__(self):
        return self.__str__()


class Case:
    def __init__(self, id_: str, attributes: List[Attribute]):
        self.attributes = {a.name: a for a in attributes}
        self.id_ = id_

    @property
    def attribute_values(self):
        return [a.value for a in self.attributes.values()]

    def __eq__(self, other):
        return self.attributes == other.attributes

    def __getitem__(self, attribute_name):
        return self.attributes.get(attribute_name, None)

    @staticmethod
    def ljust(s, sz=15):
        return str(s).ljust(sz)

    def print_values(self, all_names: Optional[List[str]] = None,
                     target: Optional[Category] = None,
                     is_corner_case: bool = False,
                     ljust_sz: int = 15):
        all_names = list(self.attributes.keys()) if not all_names else all_names
        if is_corner_case:
            case_row = self.ljust(f"corner case: ", sz=ljust_sz)
        else:
            case_row = self.ljust(f"case: ", sz=ljust_sz)
        case_row += self.ljust(self.id_, sz=ljust_sz)
        case_row += "".join([f"{self.ljust(self[name].value, sz=ljust_sz)}"
                             for name in all_names])
        if target:
            case_row += f"{self.ljust(target.name, sz=ljust_sz)}"
        print(case_row)

    def __str__(self):
        names = list(self.attributes.keys())
        ljust = max([len(name) for name in names])
        ljust = max(ljust, max([len(str(a.value)) for a in self.attributes.values()])) + 2
        row1 = f"Case {self.id_} with attributes: \n"
        row2 = [f"{name.ljust(ljust)}" for name in names]
        row2 = "".join(row2) + "\n"
        row3 = [f"{str(self.attributes[name].value).ljust(ljust)}" for name in names]
        row3 = "".join(row3)
        return row1 + row2 + row3 + "\n"

    def __repr__(self):
        return self.__str__()
