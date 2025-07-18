from __future__ import annotations
import inspect
import uuid
from dataclasses import dataclass, field

import pydot
import rustworkx as rx
from typing_extensions import Any, TYPE_CHECKING, Type, final, ClassVar, Dict

from .callable_expression import CallableExpression
from ..utils import recursive_subclasses

if TYPE_CHECKING:
    from ..rdr import RippleDownRules
    from ..rules import Rule


@dataclass
class TrackedObjectMixin:
    """
    A class that is used as a base class to all classes that needs to be tracked for RDR inference, and reasoning.
    """
    _rdr_rule: Rule = field(init=False, repr=False, hash=False, default=None)
    """
    The rule that gave this conclusion.
    """
    _rdr: RippleDownRules = field(init=False, repr=False, hash=False, default=None)
    """
    The Ripple Down Rules that classified the case and produced this conclusion.
    """
    _rdr_tracked_object_id: int = field(init=False, repr=False, default_factory=lambda: uuid.uuid4().int)
    """
    The unique identifier of the conclusion.
    """
    _dependency_graph: ClassVar[rx.PyDAG[TrackedObjectMixin]] = rx.PyDAG()
    """
    A graph that represents the relationships between all tracked objects.
    """
    _class_graph_indices: ClassVar[Dict[Type[TrackedObjectMixin], int]] = {}
    """
    The index of the current class in the dependency graph.
    """

    @classmethod
    @final
    def has(cls, tracked_object_type: Type[TrackedObjectMixin]) -> bool:
        return cls.has_one(tracked_object_type) or cls.has_many(tracked_object_type)

    @classmethod
    @final
    def has_one(cls, tracked_object_type: Type[TrackedObjectMixin]) -> bool:
        ...

    @classmethod
    @final
    def has_many(cls, tracked_object_type: Type[TrackedObjectMixin]) -> bool:
        ...

    @classmethod
    @final
    def is_a(cls, tracked_object_type: Type[TrackedObjectMixin]) -> bool:
        """
        Check if the class is a subclass of the tracked object type.
        """
        return issubclass(cls, tracked_object_type)

    @classmethod
    @final
    def depends_on(cls, tracked_object_type: Type[TrackedObjectMixin]) -> bool:
        ...

    @classmethod
    def make_class_dependency_graph(cls):
        """
        Create a direct acyclic graph containing the class hierarchy.
        """
        subclasses = recursive_subclasses(TrackedObjectMixin)
        for clazz in subclasses:
            cls._add_class_to_dependency_graph(clazz)

            bases = [base for base in clazz.__bases__ if
                     base.__module__ not in ["builtins"] and base in subclasses]

            for base in bases:
                cls._add_class_to_dependency_graph(base)
                cls._dependency_graph.add_edge(cls._class_graph_indices[clazz], cls._class_graph_indices[base], "is_a")

    @classmethod
    def to_dot(cls, filepath: str, format='png') -> None:
        if not filepath.endswith(f".{format}"):
            filepath += f".{format}"
        dot_str = cls._dependency_graph.to_dot(
            lambda node: dict(
                color='black', fillcolor='lightblue', style='filled', label=node.__name__),
            lambda edge: dict(color='black', style='solid', label=edge))
        dot = pydot.graph_from_dot_data(dot_str)[0]
        dot.write(filepath, format=format)

    @classmethod
    def _add_class_to_dependency_graph(cls, class_to_add: Type[TrackedObjectMixin]) -> None:
        """
        Add a class to the dependency graph.
        """
        if class_to_add not in cls._dependency_graph.nodes():
            cls_idx = cls._dependency_graph.add_node(class_to_add)
            cls._class_graph_indices[class_to_add] = cls_idx

    def __getattribute__(self, name: str) -> Any:
        if name not in ['_rdr_rule', '_rdr', '_rdr_tracked_object_id', '_dependency_graph', '_node_index']:
            self._record_dependency(name)
        return object.__getattribute__(self, name)

    def _record_dependency(self, attr_name):
        # Inspect stack to find instance of CallableExpression
        for frame_info in inspect.stack():
            func_name = frame_info.function
            local_self = frame_info.frame.f_locals.get("self", None)
            if (
                    func_name == "__call__" and
                    local_self is not None and
                    type(local_self) is CallableExpression
            ):
                self._used_in_tracker = True
                print("TrackedObject used inside CallableExpression")
                break
