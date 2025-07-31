from __future__ import annotations

import contextvars
import itertools
from abc import abstractmethod, ABC
from collections import defaultdict
from dataclasses import dataclass, field

from anytree import Node
from typing_extensions import Iterable, Any, Optional, Type, Dict, Callable, ClassVar
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
    parent_: Optional[SymbolicExpression] = field(init=False)
    id_: int = field(init=False, repr=False)
    node_: Node = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.id_ = id_generator(self)
        self.node_ = Node(self.name_ + f"_{self.id_}",
                         parent=self.parent_.node if self.parent_ else None)
        self.node_._expression = self

    @property
    @abstractmethod
    def name_(self) -> str:
        pass

    def all_nodes_(self) -> List[SymbolicExpression]:
        return [self] + self.descendants_

    @property
    def descendants_(self) -> List[SymbolicExpression]:
        return [d._expression for d in self.node_.descendants]

    @property
    def children_(self) -> List[SymbolicExpression]:
        return [c._expression for c in self.node_.children]

    def __getattr__(self, name):
        return Attribute(self, name)

    def __call__(self, *args, **kwargs):
        return Call(self, args, kwargs)

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
        return And([self, other])

    def __or__(self, other):
        return Or([self, other])

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


@dataclass(eq=False)
class HasDomain(SymbolicExpression, ABC):
    domain_: Iterable[Any] = field(default=None, init=False)

    def __iter__(self):
        return iter(self.domain_)

    def constrain(self, indices: Iterable[int]):
        if self.parent_ is not None and isinstance(self.parent_, HasDomain):
            self.parent_.constrain(indices)
        elif self.parent_ is None:
            self.domain_ = filter_data(self.domain_, indices)


@dataclass(eq=False)
class Variable(HasDomain):
    cls_: Type
    cls_kwargs_: Dict[str, Any] = field(default_factory=dict)
    domain_: Iterable[Any] = field(default=None, kw_only=True)
    parent_: Optional[SymbolicExpression] = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.domain_ is None:
            self.domain_: Iterable[Any] = (self.cls_(**{k: self.cls_kwargs_[k][i] for k in self.cls_kwargs_.keys()})
                                          for i in enumerate(next(iter(self.cls_kwargs_.values()), [])))

    @property
    def name_(self):
        return self.cls_.__name__

    @classmethod
    def from_domain_(cls, iterable, clazz: Optional[Type] = None,
                     parent: Optional[SymbolicExpression] = None) -> Variable:
        if in_symbolic_mode():
            if not is_iterable(iterable):
                iterable = make_list(iterable)
            if not clazz:
                clazz = type(next((iter(iterable)), None))
            return Variable(clazz, domain_=iterable, parent_=parent)
        raise TypeError(f"Method from_data of {clazz.__name__} is not usable outside RuleWriting")

    def __repr__(self):
        return (f"Symbolic({self.cls_.__name__}("
                f"{', '.join(f'{k}={v!r}' for k, v in self.cls_kwargs_.items())}))")


@dataclass(eq=False)
class Attribute(HasDomain):
    """
    A symbolic attribute that can be used to access attributes of symbolic variables.
    """
    parent_: HasDomain
    attr_name_: str

    def __post_init__(self):
        if self.domain_ is None:
            self.domain_ = (getattr(item, self.attr_name_) for item in self.parent_)

    @property
    def name_(self):
        return f"{self.parent_.name_}.{self.attr_name_}"


@dataclass(eq=False)
class Call(HasDomain):
    """
    A symbolic call that can be used to call methods on symbolic variables.
    """
    parent_: HasDomain
    args_: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs_: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.args_) > 0 and len(self.kwargs_) > 0:
            self.domain_ = (item(*self.args_, **self.kwargs_) for item in self.parent_)
        elif len(self.args_) > 0:
            self.domain_ = (item(*self.args_) for item in self.parent_)
        elif len(self.kwargs_) > 0:
            self.domain_ = (item(**self.kwargs_) for item in self.parent_)
        else:
            self.domain_ = (item() for item in self.parent_)

    @property
    def name_(self):
        return f"{self.parent_.name_}()"


@dataclass(eq=False)
class ConstrainingOperator(SymbolicExpression, ABC):
    """
    An abstract base class for operators that can constrain symbolic expressions.
    This is used to ensure that the operator can be applied to symbolic expressions
    and that it can constrain the results based on indices.
    """
    operands_indices_: Dict[HasDomain, Iterable[int]] = field(default_factory=lambda: defaultdict(list), init=False)

    @abstractmethod
    def evaluate_(self):
        """
        Evaluate the operator and set the operands indices.
        This method should be implemented by subclasses.
        """
        pass

    def constrain_(self):
        """
        Constrain the symbolic expression based on the indices.
        This method should be implemented by subclasses.
        """
        for operand, indices in self.operands_indices_.items():
            if isinstance(operand, HasDomain):
                operand.constrain(indices)
            else:
                raise TypeError(f"Operand {operand} is not a HasDomain expression.")


@dataclass(eq=False)
class UnaryOperator(ConstrainingOperator, ABC):
    """
    A base class for unary operators that can be used to apply operations on symbolic expressions.
    """
    operand_: HasDomain

    def __post_init__(self):
        if not isinstance(self.operand_, SymbolicExpression):
            self.operand_ = Variable.from_domain_(self.operand_, parent=self)
        self.evaluate_()

    @property
    def name(self):
        return f"{self.operation} {self.operand_.name}"


@dataclass(eq=False)
class Not(UnaryOperator):
    """
    A symbolic NOT operation that can be used to negate symbolic expressions.
    """

    def evaluate_(self):
        def operator_yield():
            for idx, item in enumerate(self.operand_):
                if eval(f"not item"):
                    yield idx

        self.operands_indices_[self.operand_] = operator_yield()


@dataclass(eq=False)
class BinaryOperator(ConstrainingOperator, ABC):
    """
    A base class for binary operators that can be used to combine symbolic expressions.
    """
    left_: HasDomain
    operation_: str
    right_: HasDomain

    def __post_init__(self):
        if not isinstance(self.left_, SymbolicExpression):
            self.left_ = Variable.from_domain_(self.left_, parent=self)
        if not isinstance(self.right_, SymbolicExpression):
            self.right_ = Variable.from_domain_(self.right_, parent=self)
        self.evaluate_()

    @abstractmethod
    def evaluate_(self):
        """
        Evaluate the binary operator and set the left_indices and right_indices attributes.
        This method should be implemented by subclasses.
        """
        pass

    @property
    def name_(self):
        return f"{self.left_.name_} {self.operation_} {self.right_.name_}"


@dataclass(eq=False)
class Comparator(BinaryOperator):
    """
    A symbolic equality check that can be used to compare symbolic variables.
    """

    def evaluate_(self):
        def operator_yield():
            for left_idx, left_item in enumerate(self.left_):
                for right_idx, right_item in enumerate(self.right_):
                    if eval(f"left_item {self.operation_} right_item"):
                        yield left_idx, right_idx

        data1, data2 = itertools.tee(operator_yield())
        self.operands_indices_[self.left_] = (v[0] for v in data1)
        self.operands_indices_[self.right_] = (v[1] for v in data2)


@dataclass(eq=False)
class LogicalOperator(ConstrainingOperator, ABC):
    """
    A symbolic operation that can be used to combine multiple symbolic expressions.
    """
    operands_: List[HasDomain]

    def __post_init__(self):
        if not self.operands_:
            raise ValueError("LogicalOperator requires at least one operand.")
        for i, operand in enumerate(self.operands_):
            if not isinstance(operand, SymbolicExpression):
                self.operands_[i] = Variable.from_domain_(operand, parent=self)
        self.evaluate_()

    @abstractmethod
    def evaluate_(self):
        ...

    @property
    def name_(self):
        return f" {self.__class__.__name__} ".join(operand.name_ for operand in self.operands_)


@dataclass(eq=False)
class And(LogicalOperator):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """
    def evaluate_(self):
        for operand in self.operands_:
            if isinstance(operand, ConstrainingOperator):
                operand.evaluate_()
                self.operands_indices_.update(operand.operands_indices_)
            else: # a boolean expression
                self.operands_indices_[operand] = (i for i, item in enumerate(operand) if item)


@dataclass(eq=False)
class Or(LogicalOperator):
    """
    A symbolic OR operation that can be used to combine multiple symbolic expressions.
    """

    def evaluate_(self):
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle OR logic.
        """
        if not self.operands_:
            return
        # Combine indices from all operands
        for operand in self.operands_:
            if isinstance(operand, ConstrainingOperator):
                for item, indices in operand.operands_indices_.items():
                    self.operands_indices_[item] = itertools.chain(self.operands_indices_[item], indices)
            else: # a boolean expression
                self.operands_indices_[operand] = itertools.chain(self.operands_indices_[operand],
                                                                  (i for i, item in enumerate(operand) if item))


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
                return Variable.from_domain_(clazz=symbolic_cls, iterable=args[0])
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
