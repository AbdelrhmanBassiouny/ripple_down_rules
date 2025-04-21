from __future__ import annotations

import inspect
from dataclasses import dataclass

from sqlalchemy.orm import DeclarativeBase as SQLTable
from typing_extensions import Any, Optional, Type, List, Tuple, Set, Dict, TYPE_CHECKING

from .case import create_case, Case
from ..utils import get_attribute_name, copy_case, get_hint_for_attribute, typing_to_python_type

if TYPE_CHECKING:
    from . import CallableExpression

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
    mutually_exclusive: bool = False
    """
    Whether the attribute can only take one value (i.e. True) or multiple values (i.e. False).
    """
    conditions: Optional[CallableExpression] = None
    """
    The conditions that must be satisfied for the target value to be valid.
    """
    prediction: Optional[CallableExpression] = None
    """
    The predicted value of the attribute.
    """
    scope: Optional[Dict[str, Any]] = None
    """
    The global scope of the case query. This is used to evaluate the conditions and prediction, and is what is available
    to the user when they are prompted for input. If it is not provided, it will be set to the global scope of the
    caller.
    """

    def __init__(self, case: Any, attribute_name: str,
                 target: Optional[Any] = None,
                 mutually_exclusive: bool = False,
                 conditions: Optional[CallableExpression] = None,
                 prediction: Optional[CallableExpression] = None,
                 scope: Optional[Dict[str, Any]] = None,):
        self.original_case = case
        self.case = self._get_case()

        self.attribute_name = attribute_name
        self.target = target
        self.attribute_type = self._get_attribute_type()
        self.mutually_exclusive = mutually_exclusive
        self.conditions = conditions
        self.prediction = prediction
        self.scope = scope if scope is not None else inspect.currentframe().f_back.f_globals

    def _get_case(self) -> Any:
        if not isinstance(self.original_case, (Case, SQLTable)):
            return create_case(self.original_case, max_recursion_idx=3)
        else:
            return self.original_case

    def _get_attribute_type(self) -> Type:
        """
        :return: The type of the attribute.
        """
        if self.target is not None:
            return type(self.target)
        elif hasattr(self.original_case, self.attribute_name):
            hint, origin, args = get_hint_for_attribute(self.attribute_name, self.original_case)
            if origin is not None:
                return typing_to_python_type(origin)
            elif hint is not None:
                return typing_to_python_type(hint)

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
        header = f"CaseQuery: {self.name}"
        target = f"Target: {self.name} |= {self.target if self.target is not None else '?'}"
        prediction = f"Prediction: {self.name} |= {self.prediction if self.prediction is not None else '?'}"
        conditions = f"Conditions: {self.conditions if self.conditions is not None else '?'}"
        return "\n".join([header, target, prediction, conditions])

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return CaseQuery(copy_case(self.case), self.attribute_name, self.target, self.mutually_exclusive,
                         self.conditions, self.prediction, self.scope)
