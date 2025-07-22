import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Generator, ClassVar

from typing_extensions import Type, TYPE_CHECKING, Tuple, Dict, List

from .datastructures.tracked_object import TrackedObjectMixin, Direction, Relation

if TYPE_CHECKING:
    pass


@dataclass(eq=False)
class Predicate(TrackedObjectMixin, ABC):

    @classmethod
    @abstractmethod
    def relation(cls) -> Relation:
        """
        The relation type of the predicate.
        """

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    @classmethod
    @abstractmethod
    def evaluate(cls, *args, **kwargs):
        """
        Evaluate the predicate with the given arguments.
        This method should be implemented by subclasses.
        """
        pass

    def __hash__(self):
        return hash(self.__class__.__name__)

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return False
        return self.__class__ == other.__class__


@dataclass
class IsA(Predicate):
    """
    A predicate that checks if an object type is a subclass of another object type.
    """

    @classmethod
    def relation(cls):
        return Relation.isA

    @classmethod
    def evaluate(cls, child_type: Type[TrackedObjectMixin], parent_type: Type[TrackedObjectMixin]) -> bool:
        return issubclass(child_type, parent_type)


isA = IsA()


@dataclass
class Has(Predicate):
    """
    A predicate that checks if an object type has a certain member object type.
    """

    # cached_results: ClassVar[Dict[Tuple, List]] = defaultdict(list)

    @classmethod
    def relation(cls):
        return Relation.has

    @classmethod
    # @lru_cache(maxsize=None)
    def evaluate(cls, owner_type: Type[TrackedObjectMixin],
                 member_type: Type[TrackedObjectMixin], recursive: bool = False, is_reversed: bool = False) \
            -> Generator[Tuple[Type[TrackedObjectMixin], Type[TrackedObjectMixin]], None, None]:
        if owner_type is not TrackedObjectMixin:
            direction = Direction.INBOUND.value if is_reversed else Direction.OUTBOUND.value
            owner_idx = owner_type._my_graph_idx()
            neighbors = cls._dependency_graph.adj_direction(owner_idx, direction)
            neighbor_generator = ((owner_idx, n) for n, e in neighbors.items()
                                  if (e == cls.relation() and isA(cls._dependency_graph.get_node_data(n), member_type))
                                  or (e == Relation.isA and any(cls.evaluate(cls._dependency_graph.get_node_data(n),
                                                                             member_type)))
                                  )
            latest_results = []
            for v in neighbor_generator:
                res = (cls._dependency_graph.get_node_data(v[0]), cls._dependency_graph.get_node_data(v[1]))
                latest_results.append(v)
                # cls.cached_results[(owner_type, member_type)].append(res)
                yield res
            if recursive:
                for n in [n for n, e in neighbors.items() if e == cls.relation()]:
                    for v in cls.evaluate(cls._dependency_graph.get_node_data(n), member_type, recursive=True,
                                            is_reversed=is_reversed):
                        # cls.cached_results[(owner_type, member_type)].append(v)
                        yield owner_type, v[1]
        elif member_type is not TrackedObjectMixin:
            # owner type is TrackedObjectMixin
            yield from map(lambda t: (t[1], t[0]),
                           cls.evaluate(member_type, owner_type, recursive=recursive, is_reversed=True))
        else:
            # both are TrackedObjectMixin
            yield from map(
                lambda t: (cls._dependency_graph.get_node_data(t[0]), cls._dependency_graph.get_node_data(t[1])),
                cls._edges[cls.relation()])


has = Has()


@dataclass
class DependsOn(Predicate):
    """
    A predicate that checks if an object type depends on another object type.
    """

    @classmethod
    def relation(cls):
        return Relation.dependsOn

    @classmethod
    def evaluate(cls, dependent: Type[TrackedObjectMixin],
                 dependency: Type[TrackedObjectMixin], recursive: bool = False) -> bool:
        raise NotImplementedError("Should be overridden in rdr meta")


dependsOn = DependsOn()
