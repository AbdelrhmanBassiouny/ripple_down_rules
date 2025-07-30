from __future__ import annotations

import contextvars
import itertools
from abc import abstractmethod, ABC
from dataclasses import dataclass, field

from anytree import Node
from pydantic.v1.errors import cls_kwargs
from typing_extensions import Iterable, Any, Optional, Type, Dict, Callable
from typing_extensions import dataclass_transform, List, Tuple

from .utils import is_iterable, filter_data
from .utils import make_list, IDGenerator

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


id_generator = IDGenerator()


@dataclass(eq=False)
class SymbolicExpression(ABC):
    parent: Optional[SymbolicExpression] = field(default=None, kw_only=True)
    data: Iterable[Any] = field(init=False, default_factory=list, repr=False)
    id_: int = field(init=False, repr=False)
    node: Node = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.id_ = id_generator(self)
        self.node = Node(self.name + f"_{self.id_}",
                         parent=self.parent.node if self.parent else None)
        self.node._expression = self

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def all_nodes(self) -> List[SymbolicExpression]:
        return [self] + self.descendants

    @property
    def descendants(self) -> List[SymbolicExpression]:
        return [d._expression for d in self.node.descendants]

    @property
    def children(self) -> List[SymbolicExpression]:
        return [c._expression for c in self.node.children]

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

    def __hash__(self):
        return hash(id(self))


@dataclass
class HasDomain(SymbolicExpression, ABC):
    domain: Iterable[Any] = field(default_factory=list, kw_only=True)

    def __iter__(self):
        return iter(self.domain)

    def constrain(self, indices: Iterable[int]):
        if self.parent and isinstance(self.parent, HasDomain):
            self.parent.constrain(indices)
        elif not self.parent:
            self.domain = filter_data(self.domain, indices)


@dataclass(eq=False)
class Variable(HasDomain):
    cls: Type
    cls_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.domain:
            self.domain: Iterable[Any] = (self.cls(**{k: self.cls_kwargs[k][i] for k in self.cls_kwargs.keys()})
                                          for i in enumerate(next(iter(self.cls_kwargs.values()), [])))

    @property
    def name(self):
        return self.cls.__name__

    @classmethod
    def from_domain(cls, iterable, clazz: Optional[Type] = None,
                    parent: Optional[SymbolicExpression] = None) -> Variable:
        if in_symbolic_mode():
            if not is_iterable(iterable):
                iterable = make_list(iterable)
            if not clazz:
                clazz = type(next((iter(iterable)), None))
            return Variable(clazz, domain=iterable, parent=parent)
        raise TypeError(f"Method from_data of {clazz.__name__} is not usable outside RuleWriting")

    def __repr__(self):
        return (f"Symbolic({self.cls.__name__}({', '.join(map(repr, self.args))}, "
                f"{', '.join(f'{k}={v!r}' for k, v in self.kwargs.items())}))")


@dataclass
class HasVariables:
    variables: List[Variable] = field(init=False, default_factory=list)


@dataclass(eq=False)
class Attribute(SymbolicExpression, HasDomain):
    """
    A symbolic attribute that can be used to access attributes of symbolic variables.
    """
    attr_name: str

    def __post_init__(self):
        if not self.domain:
            self.domain = (getattr(item, self.attr_name) for item in self.parent)

    @property
    def name(self):
        return f"{self.parent.name}.{self.attr_name}"


@dataclass(eq=False)
class Call(SymbolicExpression, HasDomain):
    """
    A symbolic call that can be used to call methods on symbolic variables.
    """
    args: Tuple[Any] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.args) > 0 and len(self.kwargs) > 0:
            self.domain = (item(*self.args, **self.kwargs) for item in self.parent)
        elif len(self.args) > 0:
            self.domain = (item(*self.args) for item in self.parent)
        elif len(self.kwargs) > 0:
            self.domain = (item(**self.kwargs) for item in self.parent)
        else:
            self.domain = (item() for item in self.parent)

    @property
    def name(self):
        return f"{self.parent.name}()"


@dataclass(eq=False)
class Comparator(SymbolicExpression):
    """
    A symbolic equality check that can be used to compare symbolic variables.
    """
    left: HasDomain
    operation: str
    right: HasDomain

    def __post_init__(self):
        if not isinstance(self.left, SymbolicExpression):
            self.left = Variable.from_domain(self.left, parent=self)
        if not isinstance(self.right, SymbolicExpression):
            self.right = Variable.from_domain(self.right, parent=self)

    def evaluate(self):
        def operator_yield():
            for left_idx, left_item in enumerate(self.left):
                for right_idx, right_item in enumerate(self.right):
                    if eval(f"left_item {self.operation} right_item"):
                        yield left_idx, right_idx

        data1, data2 = itertools.tee(operator_yield())
        self.left.domain_indices = (v[0] for v in data1)
        self.right.domain_indices = (v[1] for v in data2)

    @property
    def name(self):
        return f"{self.left.name} {self.operation} {self.right.name}"


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
                return Variable.from_domain(clazz=symbolic_cls, iterable=args[0])
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
