from __future__ import annotations

import itertools
from typing import dataclass_transform

from typing_extensions import Iterable, Any, Optional, Type, Dict, Callable

import contextvars

from .utils import make_list
from .utils import is_iterable

_symbolic_mode = contextvars.ContextVar("symbolic_mode", default=False)

def _set_symbolic_mode(value: bool):
    _symbolic_mode.set(value)

def in_symbolic_mode():
    return _symbolic_mode.get()


class SymbolicMode:
    def __enter__(self):
        _set_symbolic_mode(True)
        return self  # optional, depending on whether you want to assign `as` variable

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_symbolic_mode(False)


class SymbolicExpression:
    data: Iterable[Any]
    variable: Variable
    variables_data_dict: Dict[Variable, Iterable[Any]]
    parent_expression: Optional[SymbolicExpression] = None

    def __init__(self, parent_expression: Optional[SymbolicExpression] = None):
        self.data = []
        self.variables_data_dict = {}
        self.id_ = id(self)
        self.parent_expression = parent_expression
        if parent_expression is not None:
            self.variable = parent_expression.variable
        else:
            self.variable = self
        self.variables_data_dict[self.variable] = self.data

    def __getattr__(self, name):
        return Attribute(self, name)

    def __call__(self, *args, **kwargs):
        return Call(self, *args, **kwargs)

    def __eq__(self, other):
        return Comparator(self, '==', other)

    def in_(self, other):
        """
        Check if the symbolic expression is in another iterable or symbolic expression.
        """
        return in_(self, other)

    def contains_(self, item):
        """
        Check if the symbolic expression contains a specific item.
        """
        return self.__contains__(item)

    def __contains__(self, item):
        return Comparator(item, 'in', self)

    def __bool__(self):
        raise TypeError("Cannot evaluate symbolic expression to a boolean value.")

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self}>"

    def __iter__(self):
        return iter(self.data)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    def __ne__(self, other):
        return Comparator(self, '!=', other)

    def __lt__(self, other):
        return Comparator(self, '<', other)

    def __le__(self, other):
        return Comparator(self, '<=', other)

    def __gt__(self, other):
        return Comparator(self, '>', other)

    def __ge__(self, other):
        return Comparator(self, '>=', other)


class Variable(SymbolicExpression):

    def __init__(self, cls, *args, data: Optional[Iterable] = None, **kwargs):
        super().__init__()
        self.cls = cls
        self.full_name = f"{cls.__name__}"
        self.args = args
        self.kwargs = kwargs
        if data is not None:
            self.data = data
        else:
            self.data: Iterable[Any] = (cls(*(arg[i] for arg in self.args), **{k: v[i] for k, v in self.kwargs.items()})
                                        for i in range(len(self.args)))

    @classmethod
    def from_data(cls, iterable, clazz: Optional[Type] = None) -> Variable:
        if in_symbolic_mode():
            if not is_iterable(iterable):
                iterable = make_list(iterable)
            if not clazz:
                clazz = type(next((iter(iterable)), None))
            return Variable(clazz, data=iterable)
        raise TypeError(f"Method from_data of {clazz.__name__} is not usable outside RuleWriting")

    def __repr__(self):
        return (f"Symbolic({self.cls.__name__}({', '.join(map(repr, self.args))}, "
                f"{', '.join(f'{k}={v!r}' for k, v in self.kwargs.items())}))")

    def __hash__(self):
        return hash(self.id_)


class Attribute(SymbolicExpression):
    """
    A symbolic attribute that can be used to access attributes of symbolic variables.
    """

    def __init__(self, expression: SymbolicExpression, attr_name):
        super().__init__(expression)
        self.data = (getattr(item, attr_name) for item in expression)
        self.variables_data_dict[self.variable] = self.data


class Call(SymbolicExpression):
    """
    A symbolic call that can be used to call methods on symbolic variables.
    """

    def __init__(self, expression: SymbolicExpression, *args, **kwargs):
        super().__init__(expression)
        if len(args) > 0 and len(kwargs) > 0:
            self.data = (item(*args, **kwargs) for item in expression)
        elif len(args) > 0:
            self.data = (item(*args) for item in expression)
        elif len(kwargs) > 0:
            self.data = (item(**kwargs) for item in expression)
        else:
            self.data = (item() for item in expression)
        self.variables_data_dict[self.variable] = self.data

class Comparator(SymbolicExpression):
    """
    A symbolic equality check that can be used to compare symbolic variables.
    """

    def __init__(self, left: SymbolicExpression, operation: str, right: Any):
        if not isinstance(left, SymbolicExpression):
            left = Variable.from_data(left)
        if not isinstance(right, SymbolicExpression):
            right = Variable.from_data(right)
        super().__init__(left)
        self.operation = operation
        def operator_yield():
            left_idx = 0
            for left_item in left:
                right_idx = 0
                for right_item in right:
                    if eval(f"left_item {self.operation} right_item"):
                        yield left_idx, right_idx
                    right_idx += 1
                left_idx += 1
        data1, data2 = itertools.tee(operator_yield())
        self.left = left
        self.right = right
        self.variables_data_dict[self.variable] = (v[0] for v in data1)

        if isinstance(right, SymbolicExpression):
            self.variables_data_dict[right.variable] = (v[1] for v in data2)

class And(SymbolicExpression):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """

    def __init__(self, left: SymbolicExpression, right: SymbolicExpression):
        super().__init__(left)
        common_variables = set(left.variables_data_dict.keys()).intersection(right.variables_data_dict.keys())



class Mapped(SymbolicExpression):
    """
    A symbolic mapping that can be used to map symbolic variables to their attributes.
    """

    def __init__(self, expression: SymbolicExpression, mapper: Callable):
        super().__init__(expression)
        self.variables_data_dict[self.variable] = expression.data
        for item, value in self.variables_data_dict.items():
            self.variables_data_dict[item] = (mapper(v) for v in value)

@dataclass_transform()
def symbolic(cls):
    orig_new = cls.__new__ if '__new__' in cls.__dict__ else object.__new__

    def symbolic_new(symbolic_cls, *args, **kwargs):
        if in_symbolic_mode():
            if len(args) == 1 and isinstance(args[0], Iterable) and len(kwargs) == 0:
                # If the first argument is an iterable, treat it as data
                return Variable.from_data(clazz=symbolic_cls, iterable=args[0])
            return Variable(symbolic_cls, *args, **kwargs)
        return orig_new(symbolic_cls)

    cls.__new__ = symbolic_new
    return cls


def in_(item, container):
    """
    Check if the symbolic expression is in another iterable or symbolic expression.
    """
    return Comparator(item, 'in', container)


def contains(container, item):
    """
    Check if the symbolic expression contains a specific item.
    """
    return in_(item, container)