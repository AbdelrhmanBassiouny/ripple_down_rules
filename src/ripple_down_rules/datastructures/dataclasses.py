from __future__ import annotations

import inspect
from dataclasses import dataclass

from sqlalchemy.orm import DeclarativeBase as SQLTable
from typing_extensions import Any, Optional, Dict, Type

from .callable_expression import CallableExpression
from .case import create_case, Case
from ..utils import copy_case, get_case_attribute_type


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
                 attribute_type: Optional[Type] = None,
                 mutually_exclusive: Optional[bool] = None,
                 conditions: Optional[CallableExpression] = None,
                 prediction: Optional[CallableExpression] = None,
                 scope: Optional[Dict[str, Any]] = None,
                 default_value: Optional[Any] = None):
        self.original_case = case
        self.case = self._get_case()

        self.attribute_name = attribute_name
        self.target = target
        self.default_value = default_value
        if attribute_type is None:
            target_value = self.target_value
            known_value = target_value if target_value is not None else default_value
            self.attribute_type = get_case_attribute_type(self.original_case, self.attribute_name, known_value)
        else:
            self.attribute_type = attribute_type
        self.mutually_exclusive = mutually_exclusive
        self.conditions = conditions
        self.prediction = prediction
        self.scope = scope if scope is not None else inspect.currentframe().f_back.f_globals
        if self.target is not None and not isinstance(self.target, CallableExpression):
            self.target = CallableExpression(conclusion=self.target, conclusion_type=self.attribute_type,
                                             scope=self.scope)

    def _get_case(self) -> Any:
        if not isinstance(self.original_case, (Case, SQLTable)):
            return create_case(self.original_case, max_recursion_idx=3)
        else:
            return self.original_case

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

    @property
    def target_value(self) -> Any:
        """
        :return: The target value of the case query.
        """
        if isinstance(self.target, CallableExpression):
            return self.target(self.case)
        return self.target

    def __str__(self):
        header = f"CaseQuery: {self.name}"
        target = f"Target: {self.name} |= {self.target if self.target is not None else '?'}"
        prediction = f"Prediction: {self.name} |= {self.prediction if self.prediction is not None else '?'}"
        conditions = f"Conditions: {self.conditions if self.conditions is not None else '?'}"
        return "\n".join([header, target, prediction, conditions])

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        case_query_cp = CaseQuery(copy_case(self.case), self.attribute_name, target=self.target,
                                  attribute_type=self.attribute_type,
                                  mutually_exclusive=self.mutually_exclusive,
                                  conditions=self.conditions, prediction=self.prediction, scope=self.scope,
                                  default_value=self.default_value)
        case_query_cp.original_case = self.original_case
        return case_query_cp
