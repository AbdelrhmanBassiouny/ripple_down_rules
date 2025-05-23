from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass, field

import typing_extensions
from sqlalchemy.orm import DeclarativeBase as SQLTable
from typing_extensions import Any, Optional, Dict, Type, Tuple, Union, List, get_origin, Set

from .callable_expression import CallableExpression
from .case import create_case, Case
from ..utils import copy_case, make_list, make_set, get_origin_and_args_from_type_hint, get_value_type_from_type_hint, \
    typing_to_python_type


@dataclass
class CaseQuery:
    """
    This is a dataclass that represents an attribute of an object and its target value. If attribute name is
    not provided, it will be inferred from the attribute itself or from the attribute type or from the target value,
    depending on what is provided.
    """
    original_case: Any
    """
    The case that the attribute belongs to.
    """
    attribute_name: str
    """
    The name of the attribute.
    """
    _attribute_types: Tuple[Type]
    """
    The type(s) of the attribute.
    """
    mutually_exclusive: bool
    """
    Whether the attribute can only take one value (i.e. True) or multiple values (i.e. False).
    """
    _target: Optional[CallableExpression] = None
    """
    The target expression of the attribute.
    """
    default_value: Optional[Any] = None
    """
    The default value of the attribute. This is used when the target value is not provided.
    """
    scope: Optional[Dict[str, Any]] = field(default_factory=lambda: inspect.currentframe().f_back.f_back.f_globals)
    """
    The global scope of the case query. This is used to evaluate the conditions and prediction, and is what is available
    to the user when they are prompted for input. If it is not provided, it will be set to the global scope of the
    caller.
    """
    _case: Optional[Union[Case, SQLTable]] = None
    """
    The created case from the original case that the attribute belongs to.
    """
    _target_value: Optional[Any] = None
    """
    The target value of the case query. (This is the result of the target expression evaluation on the case.)
    """
    conditions: Optional[CallableExpression] = None
    """
    The conditions that must be satisfied for the target value to be valid.
    """
    is_function: bool = False
    """
    Whether the case is a dict representing the arguments of an actual function or not,
    most likely means it came from RDRDecorator, the the rdr takes function arguments and outputs the function output.
    """
    function_args_type_hints: Optional[Dict[str, Type]] = None
    """
    The type hints of the function arguments. This is used to recreate the function signature.
    """

    @property
    def case_type(self) -> Type:
        """
        :return: The type of the case that the attribute belongs to.
        """
        return self.original_case._obj_type if isinstance(self.original_case, Case) else type(self.original_case)

    @property
    def case(self) -> Any:
        """
        :return: The case that the attribute belongs to.
        """
        if self._case is not None:
            return self._case
        elif not isinstance(self.original_case, (Case, SQLTable)):
            self._case = create_case(self.original_case, max_recursion_idx=3)
        else:
            self._case = self.original_case
        return self._case

    @case.setter
    def case(self, value: Any):
        """
        Set the case that the attribute belongs to.
        """
        if not isinstance(value, (Case, SQLTable)):
            raise ValueError("The case must be a Case or SQLTable object.")
        self._case = value

    @property
    def attribute_type_hint(self) -> str:
        """
        :return: The type hint of the attribute as a typing object.
        """
        if len(self.core_attribute_type) > 1:
            attribute_types_str = f"Union[{', '.join([t.__name__ for t in self.core_attribute_type])}]"
        else:
            attribute_types_str = self.core_attribute_type[0].__name__
        if list in self.attribute_type:
            return f"List[{attribute_types_str}]"
        else:
            return attribute_types_str

    @property
    def core_attribute_type(self) -> Tuple[Type]:
        """
        :return: The core type of the attribute.
        """
        return tuple(t for t in self.attribute_type if t not in (set, list))

    @property
    def attribute_type(self) -> Tuple[Type]:
        """
        :return: The type of the attribute.
        """
        if not isinstance(self._attribute_types, tuple):
            self._attribute_types = tuple(make_set(self._attribute_types))
        origin, args = get_origin_and_args_from_type_hint(self._attribute_types)
        if origin is not None:
            att_types = make_set(origin)
            if origin in (list, set, tuple, List, Set, Union, Tuple):
                att_types.update(make_set(args))
            elif origin in (dict, Dict):
                # ignore the key type
                if args and len(args) > 1:
                    att_types.update(make_set(args[1]))
            self._attribute_types = tuple(att_types)
        if not self.mutually_exclusive and (list not in self._attribute_types):
            self._attribute_types = tuple(make_list(self._attribute_types) + [set, list])
        return self._attribute_types

    @attribute_type.setter
    def attribute_type(self, value: Type):
        """
        Set the type of the attribute.
        """
        self._attribute_types = tuple(make_list(value))

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
    def target(self) -> Optional[CallableExpression]:
        """
        :return: The target expression of the attribute.
        """
        if (self._target is not None) and (not isinstance(self._target, CallableExpression)):
            self._target = CallableExpression(conclusion=self._target, conclusion_type=self.attribute_type,
                                              scope=self.scope, mutually_exclusive=self.mutually_exclusive)
        return self._target

    @target.setter
    def target(self, value: Optional[CallableExpression]):
        """
        Set the target expression of the attribute.
        """
        if value is not None and not isinstance(value, (CallableExpression, str)):
            raise ValueError("The target must be a CallableExpression or a string.")
        self._target = value
        self.update_target_value()

    @property
    def target_value(self) -> Any:
        """
        :return: The target value of the case query.
        """
        if self._target_value is None:
            self.update_target_value()
        return self._target_value

    def update_target_value(self):
        """
        Update the target value of the case query.
        """
        if isinstance(self.target, CallableExpression):
            self._target_value = self.target(self.case)
        else:
            self._target_value = self.target

    def __str__(self):
        header = f"CaseQuery: {self.name}"
        target = f"Target: {self.name} |= {self.target if self.target is not None else '?'}"
        conditions = f"Conditions: {self.conditions if self.conditions is not None else '?'}"
        return "\n".join([header, target, conditions])

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return CaseQuery(self.original_case, self.attribute_name, self.attribute_type,
                         self.mutually_exclusive, _target=self.target, default_value=self.default_value,
                         scope=self.scope, _case=copy_case(self.case), _target_value=self.target_value,
                         conditions=self.conditions, is_function=self.is_function,
                         function_args_type_hints=self.function_args_type_hints)
