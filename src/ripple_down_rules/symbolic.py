from __future__ import annotations

import contextvars
import itertools
import weakref
from abc import abstractmethod, ABC
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field

from anytree import Node
from ordered_set import OrderedSet
from typing_extensions import Iterable, Any, Optional, Type, Dict, Set, ClassVar
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
    id_: int = field(init=False, repr=False, default=None)
    node_: Node = field(init=False, default=None, repr=False)
    id_expression_map_: ClassVar[Dict[int, SymbolicExpression]] = {}

    def __post_init__(self):
        self.id_ = id_generator(self)
        node_name = self.name_ + f"_{self.id_}"
        self.node_ = Node(node_name)
        if self.child_ is not None:
            if self.child_.node_.parent is not None:
                child_cp = self.child_.__new__(self.child_.__class__)
                child_cp.__dict__.update(self.child_.__dict__)
                child_cp.node_ = Node(self.child_.node_.name + f"_{self.id_}")
                child_cp.node_._expression = child_cp
                self.child_ = child_cp
            self.child_.node_.parent = self.node_
        self.node_._expression = self
        if self.id_ not in self.id_expression_map_:
            self.id_expression_map_[self.id_] = self


    @abstractmethod
    def evaluate_(self):
        """
        Evaluate the symbolic expression and set the operands indices.
        This method should be implemented by subclasses.
        """
        pass

    @property
    def root_(self) -> SymbolicExpression:
        """
        Get the root of the symbolic expression tree.
        """
        return self.node_.root._expression

    @property
    @abstractmethod
    def name_(self) -> str:
        pass

    @property
    def all_nodes_(self) -> List[SymbolicExpression]:
        return [self] + self.descendants_

    @property
    def all_node_names_(self) -> List[str]:
        return [node.node_.name for node in self.all_nodes_]

    @property
    def descendants_(self) -> List[SymbolicExpression]:
        return [d._expression for d in self.node_.descendants]

    @property
    def children_(self) -> List[SymbolicExpression]:
        return [c._expression for c in self.node_.children]

    def __getattr__(self, name):
        if name.startswith('_') or name in ['leaves_', 'child_']:
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

    # def __bool__(self):
    #     import pdb; pdb.set_trace()
    #     raise TypeError(f"Cannot evaluate symbolic expression {self} to a boolean value.")
        # return True

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
        # if self.id_ is None:
        # return hash(id(self))
        # else:
        return hash(self.id_)


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

    def constrain_(self, indices: Iterable[int]):
        if self.child_ is not None and isinstance(self.child_, HasDomain):
            self.child_.constrain_(indices)
        elif self.child_ is None:
            # self.domain_ = values
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

    def evaluate_(self):
        # This method is intentionally left empty as Variable does not perform any evaluation.
        # Variables are leaves in the symbolic expression tree and their domains are set during initialization.
        pass

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


    def evaluate_(self):
        self.child_.evaluate_()
        if self.domain_ is None:
            self.domain_ = (getattr(item, self.attr_name_) for item in self.child_)
        if self.root_ is self:
            leaf_id = self.leaves_.pop().id_
            self.id_expression_map_[leaf_id].constrain_(OrderedSet(i for i, v in enumerate(self) if v))

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


    def evaluate_(self):
        self.child_.evaluate_()
        if len(self.args_) > 0 and len(self.kwargs_) > 0:
            self.domain_ = [item(*self.args_, **self.kwargs_) for item in self.child_]
        elif len(self.args_) > 0:
            self.domain_ = [item(*self.args_) for item in self.child_]
        elif len(self.kwargs_) > 0:
            self.domain_ = [item(**self.kwargs_) for item in self.child_]
        else:
            self.domain_ = [item() for item in self.child_]
        if self.root_ is self:
            leaf_id = self.leaves_.pop().id_
            self.id_expression_map_[leaf_id].constrain_(OrderedSet(i for i, v in enumerate(self) if v))

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
    operands_indices_: Dict[int, OrderedSet[int]] = field(default_factory=lambda: defaultdict(OrderedSet), init=False)


    def constrain_(self):
        """
        Constrain the symbolic expression based on the indices.
        This method should be implemented by subclasses.
        """
        for operand_id, indices in self.operands_indices_.items():
            self.id_expression_map_[operand_id].constrain_(indices)

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
    def name_(self):
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
        return self.operation_

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
            self.left_.evaluate_()
            self.right_.evaluate_()
            for left_idx, left_item in enumerate(self.left_):
                for right_idx, right_item in enumerate(self.right_):
                    if eval(f"left_item {self.operation_} right_item"):
                        yield left_idx, right_idx

        data = list(operator_yield())
        self.operands_indices_[self.left_.leaves_.pop().id_] = OrderedSet(v[0] for v in data)
        self.operands_indices_[self.right_.leaves_.pop().id_] = OrderedSet(v[1] for v in data)

        if self.root_ is self:
            for item_id, indices in self.operands_indices_.items():
                self.id_expression_map_[item_id].constrain_(indices)


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
        return self.__class__.__name__

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
            operand.evaluate_()
            if isinstance(operand, ConstrainingOperator):
                operand.constrain_()
                self.operands_indices_.update(operand.operands_indices_)
            elif isinstance(operand, HasDomain):  # a boolean expression
                leaf = operand.leaves_.pop()
                self.operands_indices_[leaf.id_] = OrderedSet(i for i, v in enumerate(operand) if v)
                operand.constrain_(self.operands_indices_[leaf.id_])
            else:
                raise TypeError(f"Operand {operand} is neither a ConstrainingOperator nor a HasDomain expression.")


@dataclass(eq=False)
class Or(LogicalOperator):
    """
    A symbolic OR operation that can be used to combine multiple symbolic expressions.
    """
    _leaves_replacements: Dict[HasDomain, HasDomain] = field(init=False, default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        # Find common leaves between operands and split them into separate leaves, each connected to a separate operand.
        # This is necessary to ensure that the leaves of the OR operator are not shared between operands, which would
        # make the evaluation of each operand affect the others. Instead, we want each operand to have a copy of the
        # leaves, so that they can be evaluated independently. The leaves here are the symbolic variables that will be
        # constrained by the OR operator.
        all_leaves = [operand.leaves_ for operand in self.operands_]
        shared_leaves = set.intersection(*all_leaves)
        for leaf in shared_leaves:
            first_occurrence = True
            for operand_leaves in all_leaves:
                if leaf in operand_leaves:
                    if first_occurrence:
                        first_occurrence = False
                        continue
                    leaf.domain_ = copy(leaf.domain_)

    def evaluate_(self):
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle OR logic.
        """
        # Combine indices from all operands
        for operand in self.operands_:
            operand.evaluate_()
            if isinstance(operand, ConstrainingOperator):
                for item_id, indices in operand.operands_indices_.items():
                    self.operands_indices_[item_id].update(indices)
            else:  # a boolean expression
                leaf = operand.leaves_.pop()
                self.operands_indices_[leaf.id_].update(OrderedSet(i for i, v in enumerate(leaf) if v))
        if self.root_ is self:
            for operand_id, indices in self.operands_indices_.items():
                self.id_expression_map_[operand_id].constrain_(indices)

    def replace_leaf(self, old_leaf, new_leaf):
        self._leaves_replacements[old_leaf] = new_leaf


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
