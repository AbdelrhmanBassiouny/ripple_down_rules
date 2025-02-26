from enum import auto, Enum

from typing_extensions import List


class ExpressionParser(Enum):
    """
    Parsers for expressions to evaluate and encapsulate the expression into a callable function.
    """
    ASTVisitor: auto()
    """
    Generic python Abstract Syntax Tree that detects variables, attributes, binary/boolean expressions , ...etc.
    """
    Alchemy: auto()
    """
    Specific for SQLAlchemy expressions on ORM Tables.
    """


class PromptFor(Enum):
    """
    The reason of the prompt. (e.g. get conditions, or conclusions).
    """
    Conditions: str = "conditions"
    """
    Prompt for rule conditions about a case.
    """
    Conclusion: str = "conclusion"
    """
    Prompt for rule conclusion about a case.
    """

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


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
