import os.path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os.path import dirname

from typing_extensions import Type, ClassVar, TYPE_CHECKING

from .rdr_decorators import RDRDecorator

if TYPE_CHECKING:
    from .datastructures.tracked_object import TrackedObjectMixin


@dataclass
class Predicate(ABC):
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


@dataclass
class IsA(Predicate):
    """
    A predicate that checks if an object is of a certain type.
    """

    @classmethod
    def evaluate(cls, tracked_object_type: Type[TrackedObjectMixin]) -> bool:
        return issubclass(cls, tracked_object_type)


@dataclass
class Has(Predicate):
    """
    A predicate that checks if an object has a certain type.
    """

    @classmethod
    def evaluate(cls, tracked_object_type: Type[TrackedObjectMixin], recursive: bool = False) -> bool:
        pass