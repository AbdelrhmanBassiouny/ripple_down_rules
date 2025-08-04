from typing import TypeVar, Type

from typing_extensions import Any, Optional, Union, Iterable

from . import symbolic
from .symbolic import Variable, and_, SymbolicExpression, LogicalOperator, Comparator, ConstrainingOperator, Or
from .utils import render_tree

T = TypeVar('T')  # Define type variable "T"


def entity(entity_var: T, *properties: SymbolicExpression) -> T:
    for prop in properties:
        render_tree(prop.node_.root, True, "query_tree", view=True, use_legend=False)
        prop.root_.evaluate_()
    return iter((v for _, v in entity_var))


def an(entity_type: Type[T], domain: Optional[Any] = None) -> Union[T, Iterable[T]]:
    return symbolic.Variable.from_domain_((v for v in domain if isinstance(v, entity_type)), clazz=entity_type)
