from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass

from sqlalchemy.orm import DeclarativeBase as SQLTable
from typing_extensions import Any, Optional, Type, Union

from .table import create_row, Case
from ..utils import get_attribute_name, copy_orm_instance_with_relationships, copy_case


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
    attribute: Optional[Any] = None
    """
    The attribute itself.
    """
    targets: Optional[Any] = None
    """
    The target value of the attribute.
    """
    attribute_name: Optional[str] = None
    """
    The name of the attribute.
    """
    attribute_type: Optional[Type] = None
    """
    The type of the attribute.
    """
    relational_representation: Optional[str] = None
    """
    The representation of the target value in relational form.
    """

    def __init__(self, case: Any, attribute: Optional[Any] = None, target: Optional[Any] = None,
                 attribute_name: Optional[str] = None, attribute_type: Optional[Type] = None,
                 relational_representation: Optional[str] = None):

        if attribute_name is None:
            attribute_name = get_attribute_name(case, attribute, attribute_type, target)
        self.attribute_name = attribute_name

        if not isinstance(case, (Case, SQLTable)):
            case = create_row(case, max_recursion_idx=3)
        self.case = case

        self.attribute = getattr(self.case, self.attribute_name) if self.attribute_name else None
        self.attribute_type = type(self.attribute) if self.attribute else None
        self.target = target
        self.relational_representation = relational_representation

    @property
    def name(self):
        return self.attribute_name if self.attribute_name else self.__class__.__name__

    def __str__(self):
        if self.relational_representation:
            return f"{self.name} |= {self.relational_representation}"
        else:
            return f"{self.target}"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return CaseQuery(copy_case(self.case), attribute_name=self.attribute_name, target=self.target)
