from __future__ import annotations

import contextvars
import itertools
import weakref
from abc import abstractmethod, ABC
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field

from anytree import Node
from typing_extensions import Iterable, Any, Optional, Type, Dict, Set, ClassVar, Callable
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


@dataclass
class HashedValue:
    value: Any
    id_: Optional[int] = field(default=None, repr=False)

    def __post_init__(self):
        if self.id_ is None:
            self.id_ = id(self.value)

    def __hash__(self):
        return hash(self.id_)


@dataclass
class HashedIterable:
    """
    A wrapper for an iterable that hashes its items.
    This is useful for ensuring that the items in the iterable are unique and can be used as keys in a dictionary.
    """
    values: Dict[int, Any] = field(default_factory=dict)

    @classmethod
    def from_iterable(cls, iterable: Iterable[Any]) -> HashedIterable:
        """
        Create a HashedIterable from an iterable.

        :param iterable: An iterable of values to hash.
        :return: A new HashedIterable instance.
        """
        if not isinstance(iterable, HashedIterable):
            return cls({k: v for k, v in enumerate(iterable)})
        return iterable

    @property
    def ids(self) -> Set[int]:
        """
        Get the ids of the hashed values.

        :return: A set of ids of the hashed values.
        """
        return set(self.values.keys())

    def map(self, func: Callable[[Any], Any]):
        """
        Apply a function to each value in the HashedIterable and return a new HashedIterable.

        :param func: The function to apply to each value.
        :return: A new HashedIterable with the transformed values.
        """
        return HashedIterable({k: func(v) for k, v in self.values.items()})

    def filter(self, selected_ids: Iterable[int]):
        """
        Filter the HashedIterable based on a set of selected ids.

        :param selected_ids: An iterable of ids to keep in the HashedIterable.
        :return: A new HashedIterable containing only the items with the specified ids.
        """
        self.values = {k: self.values[k] for k in selected_ids}
        return HashedIterable(self.values)

    def update(self, values: HashedIterable):
        """
        Update the hashed values with another HashedIterable.

        :param values: The HashedIterable to update with.
        """
        self.values.update(values.values)

    def add(self, value: HashedValue):
        """
        Add a HashedValue to the hashed values.

        :param value: The HashedValue to add.
        """
        if value.id_ not in self.values:
            self.values[value.id_] = value
        else:
            raise ValueError(f"Value with id {value.id_} already exists in the hashed values.")

    def union(self, other: HashedIterable) -> HashedIterable:
        """
        Create a union of two HashedIterables.

        :param other: The other HashedIterable to union with.
        :return: A new HashedIterable containing the union of both.
        """
        all_keys = self.values.keys() | other.values.keys()
        return HashedIterable({k: self.values.get(k, other.values[k]) for k in all_keys})

    def intersection(self, other: HashedIterable) -> HashedIterable:
        common_keys = self.values.keys() & other.values.keys()
        return HashedIterable({k: self.values[k] for k in common_keys})

    def __iter__(self):
        """
        Iterate over the hashed values.

        :return: An iterator over the hashed values.
        """
        return iter(self.values.items())

    def __getitem__(self, id_: int) -> HashedValue:
        """
        Get the HashedValue by its id.

        :param id_: The id of the HashedValue to get.
        :return: The HashedValue with the given id.
        """
        return self.values[id_]

    def __copy__(self):
        """
        Create a shallow copy of the HashedIterable.

        :return: A new HashedIterable instance with the same values.
        """
        return HashedIterable(values=self.values.copy())

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
        if name.startswith('_') or name in ['leaves_', 'child_', 'all_leaf_instances_']:
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
    domain_: HashedIterable = field(default=None, init=False)
    # _domain_manager: TeeManager = field(init=False, default=None)
    
    def __post_init__(self):
        if self.domain_ is not None and not isinstance(self.domain_, HashedIterable):
            self.domain_ = HashedIterable.from_iterable(self.domain_)
        super().__post_init__()

    # def evaluate_(self):
    #     if self.domain_ is not None :
    #         self.domain_ = HashedIterable.from_iterable(self.domain_)

    def __iter__(self):
        return iter(self.domain_)

    # def get_domain_(self):
    #     if self._domain_manager is None:
    #         self._domain_manager = TeeManager(self.domain_)
    #     return self._domain_manager.get_iterator()

    def constrain_(self, ids: Iterable[int]):
        if self.child_ is not None and isinstance(self.child_, HasDomain):
            self.child_.constrain_(ids)
        elif self.child_ is None:
            self.domain_.filter(ids)

    @property
    def leaves_(self) -> Set[HasDomain]:
        child_leaves = set()
        if self.child_ is not None and hasattr(self.child_, 'leaves_'):
            return self.child_.leaves_
        else:
            return {self}

    @property
    def all_leaf_instances_(self) -> List[HasDomain]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        child_leaves = []
        if self.child_ is not None and hasattr(self.child_, 'all_leaf_instances_'):
            child_leaves = self.child_.all_leaf_instances_
        return [self] + child_leaves


@dataclass(eq=False)
class Variable(HasDomain):
    cls_: Optional[Type] = field(default=None)
    cls_kwargs_: Dict[str, Any] = field(default_factory=dict)
    domain_: HashedIterable = field(default=None, kw_only=True)
    child_: Optional[SymbolicExpression] = field(default=None, kw_only=True)

    def evaluate_(self):
        if self.domain_ is None and self.cls is not None:
            domain_values = (self.cls_(**{k: self.cls_kwargs_[k][i] for k in self.cls_kwargs_.keys()})
                             for i in enumerate(next(iter(self.cls_kwargs_.values()), [])))
            self.domain_: HashedIterable = HashedIterable.from_iterable(domain_values)

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
            self.domain_ = self.child_.domain_.map(lambda v: getattr(v, self.attr_name_))
        leaf = self.leaves_.pop()
        if self.root_ is self:
            leaf.domain_.filter([id_ for id_, value in self if value])

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
            self.domain_ = self.child_.domain_.map(lambda v: v(*self.args_, **self.kwargs_))
        elif len(self.args_) > 0:
            self.domain_ = self.child_.domain_.map(lambda v: v(*self.args_))
        elif len(self.kwargs_) > 0:
            self.domain_ = self.child_.domain_.map(lambda v: v(**self.kwargs_))
        else:
            self.domain_ = self.child_.domain_.map(lambda v: v())
        leaf = self.leaves_.pop()
        if self.root_ is self:
            leaf.domain_.filter([id_ for id_, value in self if value])

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
    operands_values_: Dict[int, HashedIterable] = field(default_factory=lambda: defaultdict(HashedIterable), init=False)


    def constrain_(self):
        """
        Constrain the symbolic expression based on the indices.
        This method should be implemented by subclasses.
        """
        for operand_id, values in self.operands_values_.items():
            self.id_expression_map_[operand_id].constrain_(values.ids)

    @property
    @abstractmethod
    def leaves_(self) -> Set[HasDomain]:
        """
        :return: Set of leaves of symbolic expressions, these are the variables that will have their domains constrained.
        """
        ...

    @property
    @abstractmethod
    def all_leaf_instances_(self) -> List[HasDomain]:
        """
        :return: List of all leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
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

    @property
    def all_leaf_instances_(self) -> List[HasDomain]:
        return self.operand_.all_leaf_instances_


@dataclass(eq=False)
class Not(UnaryOperator):
    """
    A symbolic NOT operation that can be used to negate symbolic expressions.
    """

    def evaluate_(self):
        def operator_yield():
            yield from (id_ for id_, value in self.operand_ if not value)

        operand_leaf = self.operand_.leaves_.pop()
        operand_leaf.domain_.filter(operator_yield())


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

    @property
    def all_leaf_instances_(self) -> List[HasDomain]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        return self.left_.all_leaf_instances_ + self.right_.all_leaf_instances_


@dataclass(eq=False)
class Comparator(BinaryOperator):
    """
    A symbolic equality check that can be used to compare symbolic variables.
    """

    def evaluate_(self):
        def operator_yield():
            self.left_.evaluate_()
            self.right_.evaluate_()
            for left_id, left_value in self.left_:
                for right_id, right_value in self.right_:
                    if eval(f"left_value {self.operation_} right_value"):
                        yield left_id, right_id

        data = list(operator_yield())
        for i, operand in enumerate([self.left_, self.right_]):
            # operand_leaf = operand.leaves_.pop()
            operand.constrain_([v[i] for v in data])

        # if self.root_ is self:
        #     for item_id, values in self.operands_values_.items():
        #         self.id_expression_map_[item_id].constrain_(values)


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

    @property
    def all_leaf_instances_(self) -> List[HasDomain]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        all_leaves = []
        for operand in self.operands_:
            all_leaves.extend(operand.all_leaf_instances_)
        return all_leaves


@dataclass(eq=False)
class And(LogicalOperator):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """

    def evaluate_(self):
        for operand in self.operands_:
            operand.evaluate_()
            # if isinstance(operand, ConstrainingOperator):
            #     operand.constrain_()
                # self.operands_values_.update(operand.operands_values_)
            if not isinstance(operand, ConstrainingOperator):
                if isinstance(operand, HasDomain):  # a boolean expression
                    leaf = operand.leaves_.pop()
                    operand.constrain_([id_ for id_, value in operand if value])
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
        all_leaves = [operand.all_leaf_instances_ for operand in self.operands_]
        shared_leaves = set.intersection(*[set(operand_leaves) for operand_leaves in all_leaves])
        for leaf in shared_leaves:
            first_occurrence = True
            for operand_leaves in all_leaves:
                if leaf in operand_leaves:
                    if first_occurrence:
                        first_occurrence = False
                        continue
                    leaf_instances = [l for l in operand_leaves if l.id_ == leaf.id_]
                    leaf_instances[0].domain_ = copy(leaf.domain_)
                    for leaf_instance in leaf_instances[1:]:
                        leaf_instance.domain_ = leaf_instances[0].domain_

    def evaluate_(self):
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle OR logic.
        """
        # Combine indices from all operands
        for operand in self.operands_:
            operand.evaluate_()
            if isinstance(operand, ConstrainingOperator):
                for leaf in operand.leaves_:
                    self.operands_values_[leaf.id_].update(leaf.domain_)
            else:  # a boolean expression
                leaf = operand.leaves_.pop()
                self.operands_values_[leaf.id_].update(leaf.domain_.filter(i for i, v in enumerate(leaf) if v))
        for operand_id, values in self.operands_values_.items():
            self.id_expression_map_[operand_id].domain_ = values

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
