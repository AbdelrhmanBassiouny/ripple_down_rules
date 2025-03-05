from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Optional


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
