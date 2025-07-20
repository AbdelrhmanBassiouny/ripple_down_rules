import os.path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from os.path import dirname

from typing_extensions import Type, ClassVar, TYPE_CHECKING

from .datastructures.tracked_object import TrackedObjectMixin, Direction, Relation


@dataclass
class Predicate(TrackedObjectMixin, ABC):
    models_dir: ClassVar[str] = os.path.join(dirname(__file__), "predicates_models")

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


@dataclass(eq=False)
class IsA(Predicate):
    """
    A predicate that checks if an object type is a subclass of another object type.
    """

    @classmethod
    def evaluate(cls, child_type: Type[TrackedObjectMixin], parent_type: Type[TrackedObjectMixin]) -> bool:
        return issubclass(child_type, parent_type)

isA = IsA()


@dataclass(eq=False)
class Has(Predicate):
    """
    A predicate that checks if an object type has a certain member object type.
    """

    @classmethod
    def evaluate(cls, owner_type: Type[TrackedObjectMixin],
                 member_type: Type[TrackedObjectMixin], recursive: bool = False) -> bool:
        neighbors = cls._dependency_graph.adj_direction(owner_type._my_graph_idx(), Direction.OUTBOUND.value)
        curr_val = any(e == Relation.has and isA(cls._dependency_graph.get_node_data(n), member_type)
                       or e == Relation.isA and cls.evaluate(cls._dependency_graph.get_node_data(n), member_type)
                       for n, e in neighbors.items())
        if recursive:
            return curr_val or any((e == Relation.has
                                    and cls.evaluate(cls._dependency_graph.get_node_data(n), member_type, recursive=True))
                                   for n, e in neighbors.items())
        else:
            return curr_val

has = Has()


@dataclass(eq=False)
class DependsOn(Predicate):
    """
    A predicate that checks if an object type depends on another object type.
    """

    @classmethod
    def evaluate(cls, dependent: Type[TrackedObjectMixin],
                 dependency: Type[TrackedObjectMixin], recursive: bool = False) -> bool:
        raise NotImplementedError("Should be overridden in rdr meta")


dependsOn = DependsOn()