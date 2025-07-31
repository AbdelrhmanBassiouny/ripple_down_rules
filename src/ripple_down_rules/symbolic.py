from __future__ import annotations

import contextvars
import itertools
import weakref
from abc import abstractmethod, ABC
from collections import defaultdict
from dataclasses import dataclass, field

from anytree import Node
from typing_extensions import Iterable, Any, Optional, Type, Dict, Set
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
    child_: Optional[SymbolicExpression] = field(init=False)
    id_: int = field(init=False, repr=False)
    node_: Node = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.id_ = id_generator(self)
        self.node_ = Node(self.name_ + f"_{self.id_}")
        if self.child_ is not None:
            self.child_.node_.parent = self.node_
        self.node_._expression = self

    @property
    @abstractmethod
    def name_(self) -> str:
        pass

    @property
    def all_nodes_(self) -> List[SymbolicExpression]:
        return [self] + self.descendants_

    @property
    def descendants_(self) -> List[SymbolicExpression]:
        return [d._expression for d in self.node_.descendants]

    @property
    def children_(self) -> List[SymbolicExpression]:
        return [c._expression for c in self.node_.children]

    def __getattr__(self, name):
        if name in ['roots_', 'child_']:
            raise AttributeError(name)
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


class _ManagedTeeIterator:
    def __init__(self, base_iter, on_close, iterator_id):
        self._iter = base_iter
        self._on_close = on_close
        self._id = iterator_id

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def close(self):
        self._on_close(self._id)

    def __del__(self):
        self.close()


class TeeManager:
    def __init__(self, source_iterable):
        self._source = iter(source_iterable)
        self._tee_root = itertools.tee(self._source, 1)[0]
        self._iterators = []  # List of (weakref to iterator, id)

    def get_iterator(self):
        new_iter, self._tee_root = itertools.tee(self._tee_root, 2)
        # Store weakref so it doesn't prevent garbage collection
        ref = weakref.ref(new_iter)
        self._iterators.append((ref, id(new_iter)))
        return _ManagedTeeIterator(new_iter, self._unregister, id(new_iter))

    def _unregister(self, iterator_id):
        self._iterators = [(ref, iid) for (ref, iid) in self._iterators if iid != iterator_id]

    def cleanup_dead_iterators(self):
        # Optional: prune dead iterators
        self._iterators = [(ref, iid) for (ref, iid) in self._iterators if ref() is not None]


@dataclass(eq=False)
class HasDomain(SymbolicExpression, ABC):
    domain_: Iterable[Any] = field(default=None, init=False)
    # _domain_manager: TeeManager = field(init=False, default=None)

    def __iter__(self):
        return iter(self.domain_)

    # def get_domain_(self):
    #     if self._domain_manager is None:
    #         self._domain_manager = TeeManager(self.domain_)
    #     return self._domain_manager.get_iterator()

    def constrain(self, indices: Iterable[int]):
        if self.child_ is not None and isinstance(self.child_, HasDomain):
            self.child_.constrain(indices)
        elif self.child_ is None:
            self.domain_ = filter_data(self.domain_, indices)

    @property
    def leaves_(self) -> Set[HasDomain]:
        if self.child_ is not None and hasattr(self.child_, 'leaves_'):
            return self.child_.leaves_
        else:
            return {self}


@dataclass(eq=False)
class Variable(HasDomain):
    cls_: Optional[Type] = field(default=None)
    cls_kwargs_: Dict[str, Any] = field(default_factory=dict)
    domain_: Iterable[Any] = field(default=None, kw_only=True)
    child_: Optional[SymbolicExpression] = field(default=None, kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if self.domain_ is None and self.cls is not None:
            self.domain_: Iterable[Any] = (self.cls_(**{k: self.cls_kwargs_[k][i] for k in self.cls_kwargs_.keys()})
                                           for i in enumerate(next(iter(self.cls_kwargs_.values()), [])))

    @property
    def name_(self):
        return self.cls_.__name__

    @classmethod
    def from_domain_(cls, iterable, clazz: Optional[Type] = None,
                     child: Optional[SymbolicExpression] = None) -> Variable:
        if in_symbolic_mode():
            if not is_iterable(iterable):
                iterable = make_list(iterable)
            if not clazz:
                clazz = type(next((iter(iterable)), None))
            return Variable(clazz, domain_=iterable, child_=child)
        raise TypeError(f"Method from_data of {clazz.__name__} is not usable outside RuleWriting")

    def __repr__(self):
        return (f"Symbolic({self.cls_.__name__}("
                f"{', '.join(f'{k}={v!r}' for k, v in self.cls_kwargs_.items())}))")


@dataclass(eq=False)
class Attribute(HasDomain):
    """
    A symbolic attribute that can be used to access attributes of symbolic variables.
    """
    child_: HasDomain
    attr_name_: str

    def __post_init__(self):
        super().__post_init__()
        if self.domain_ is None:
            self.domain_ = (getattr(item, self.attr_name_) for item in self.child_)

    @property
    def name_(self):
        return f"{self.child_.name_}.{self.attr_name_}"

    @property
    def leaves_(self) -> Set[HasDomain]:
        return self.child_.leaves_


@dataclass(eq=False)
class Call(HasDomain):
    """
    A symbolic call that can be used to call methods on symbolic variables.
    """
    child_: HasDomain
    args_: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs_: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if len(self.args_) > 0 and len(self.kwargs_) > 0:
            self.domain_ = [item(*self.args_, **self.kwargs_) for item in self.child_]
        elif len(self.args_) > 0:
            self.domain_ = [item(*self.args_) for item in self.child_]
        elif len(self.kwargs_) > 0:
            self.domain_ = [item(**self.kwargs_) for item in self.child_]
        else:
            self.domain_ = [item() for item in self.child_]

    @property
    def name_(self):
        return f"{self.child_.name_}()"

    @property
    def leaves_(self) -> Set[HasDomain]:
        return self.child_.leaves_


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

    @property
    @abstractmethod
    def leaves_(self) -> Set[HasDomain]:
        """
        :return: Set of leaves of symbolic expressions, these are the variables that will have their domains constrained.
        """
        ...


@dataclass(eq=False)
class UnaryOperator(ConstrainingOperator, ABC):
    """
    A base class for unary operators that can be used to apply operations on symbolic expressions.
    """
    operand_: HasDomain

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.operand_, SymbolicExpression):
            self.operand_ = Variable.from_domain_(self.operand_)

    @property
    def name(self):
        return f"{self.operation} {self.operand_.name}"

    @property
    def leaves_(self) -> Set[HasDomain]:
        return self.operand_.leaves_


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
    child_: SymbolicExpression = field(init=False, default=None)

    def __post_init__(self):
        if not isinstance(self.left_, SymbolicExpression):
            self.left_ = Variable.from_domain_(self.left_)
        if not isinstance(self.right_, SymbolicExpression):
            self.right_ = Variable.from_domain_(self.right_)
        super().__post_init__()
        self.left_.node_.parent = self.node_
        self.right_.node_.parent = self.node_

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

    @property
    def leaves_(self) -> Set[HasDomain]:
        return self.left_.leaves_.union(self.right_.leaves_)


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

        data = list(operator_yield())
        self.operands_indices_[self.left_.leaves_.pop()] = [v[0] for v in data]
        self.operands_indices_[self.right_.leaves_.pop()] = [v[1] for v in data]


@dataclass(eq=False)
class LogicalOperator(ConstrainingOperator, ABC):
    """
    A symbolic operation that can be used to combine multiple symbolic expressions.
    """
    operands_: List[HasDomain]
    child_: SymbolicExpression = field(init=False, default=None)

    def __post_init__(self):
        for operand in self.operands_:
            if not isinstance(operand, SymbolicExpression):
                self.operands_ = Variable.from_domain_(operand)
        super().__post_init__()
        for operand in self.operands_:
            operand.node_.parent = self.node_

    @abstractmethod
    def evaluate_(self):
        ...

    @property
    def name_(self):
        return f" {self.__class__.__name__} ".join(operand.name_ for operand in self.operands_)

    @property
    def leaves_(self) -> Set[HasDomain]:
        leaves = set()
        for operand in self.operands_:
             leaves.update(operand.leaves_)
        return leaves


@dataclass(eq=False)
class And(LogicalOperator):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """

    def evaluate_(self):
        for operand in self.operands_:
            if isinstance(operand, ConstrainingOperator):
                # operand.constrain_()
                self.operands_indices_.update(operand.operands_indices_)
            else:  # a boolean expression
                self.operands_indices_[operand] = (i for i, item in enumerate(operand) if item)
                # operand.constrain(self.operands_indices_[operand])


@dataclass(eq=False)
class Or(LogicalOperator):
    """
    A symbolic OR operation that can be used to combine multiple symbolic expressions.
    """
    _leaves_replacements: Dict[HasDomain, HasDomain] = field(init=False, default_factory=dict)

    def evaluate_(self):
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle OR logic.
        """
        # Combine indices from all operands
        for operand in self.operands_:
            if isinstance(operand, ConstrainingOperator):
                for item, indices in operand.operands_indices_.items():
                    self.operands_indices_[item] = itertools.chain(self.operands_indices_[item], indices)
            else:  # a boolean expression
                self.operands_indices_[operand.leaves_.pop()] = itertools.chain(self.operands_indices_[operand.leaves_.pop()],
                                                                                (i for i, item in enumerate(operand) if item))
        # for operand, indices in self.operands_indices_.items():
        #     operand.constrain(self.operands_indices_[operand])

    def replace_leaf(self, old_leaf, new_leaf):
        self._roots_replacements[old_leaf] = new_leaf


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


def and_(*conditions):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """
    return And(list(conditions))


def or_(*conditions):
    """
    A symbolic OR operation that can be used to combine multiple symbolic expressions.
    """
    return Or(list(conditions))


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
