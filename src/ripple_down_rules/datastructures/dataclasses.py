from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import DeclarativeBase as SQLTable
from typing_extensions import Any, Optional, Type, List, Tuple, Set, Dict

from .case import create_case, Case
from ..utils import get_attribute_name, copy_case, get_hint_for_attribute, typing_to_python_type


@dataclass
class CaseQuery:
    """
    This is a dataclass that represents an attribute of an object and its target value. If attribute name is
    not provided, it will be inferred from the attribute itself or from the attribute type or from the target value,
    depending on what is provided.
    """
    case: Any
    """
    The case that the attribute belongs to.
    """
    attribute_name: str
    """
    The name of the attribute.
    """
    target: Optional[Any] = None
    """
    The target value of the attribute.
    """
    relational_representation: Optional[str] = None
    """
    The representation of the target value in relational form.
    """

    def __init__(self, case: Any, attribute_name: str,
                 target: Optional[Any] = None,
                 relational_representation: Optional[str] = None):
        self.case = case
        self.attribute_name = attribute_name

        self.attribute_type = None
        if target is not None:
            self.attribute_type = type(target)
        elif hasattr(case, attribute_name):
            hint, origin, args = get_hint_for_attribute(attribute_name, case)
            if origin is not None:
                self.attribute_type = typing_to_python_type(origin)
            elif hint is not None:
                self.attribute_type = typing_to_python_type(hint)

        self.target = target
        if not isinstance(case, (Case, SQLTable)):
            self.case = create_case(case, max_recursion_idx=3)
        self.relational_representation = relational_representation

    @property
    def name(self):
        """
        :return: The name of the case query.
        """
        return f"{self.case_name}.{self.attribute_name}"

    @property
    def case_name(self) -> str:
        """
        :return: The name of the case.
        """
        return self.case._name if isinstance(self.case, Case) else self.case.__class__.__name__

    def __str__(self):
        if self.relational_representation:
            return f"{self.name} |= {self.relational_representation}"
        else:
            return f"{self.name} = {self.target}"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return CaseQuery(copy_case(self.case), self.attribute_name, target=self.target,
                         relational_representation=self.relational_representation)
