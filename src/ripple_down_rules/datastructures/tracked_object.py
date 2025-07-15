from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass, field, Field, fields
from enum import Enum
from functools import lru_cache

import pydot
import rustworkx as rx
from typing_extensions import Any, TYPE_CHECKING, Type, final, ClassVar, Dict, List, Optional, Tuple

from .field_info import FieldInfo
from .. import logger
from ..utils import recursive_subclasses

if TYPE_CHECKING:
    from ..rdr import RippleDownRules
    from ..rules import Rule


class Direction(Enum):
    OUTBOUND = False
    INBOUND = True


class Relation(str, Enum):
    has = "has"
    isA = "isA"
    dependsOn = "dependsOn"


@dataclass(unsafe_hash=True)
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
    _rdr_tracked_object_id: int = field(init=False, repr=False, hash=False,
                                        compare=False, default_factory=lambda: uuid.uuid4().int)
    """
    The unique identifier of the conclusion.
    """
    _dependency_graph: ClassVar[rx.PyDAG[Type[TrackedObjectMixin]]] = rx.PyDAG()
    """
    A graph that represents the relationships between all tracked objects.
    """
    _class_graph_indices: ClassVar[Dict[Type[TrackedObjectMixin], int]] = {}
    """
    The index of the current class in the dependency graph.
    """
    _composition_edges: ClassVar[List[Tuple[Type[TrackedObjectMixin], ...]]] = []
    """
    The edges that represent composition relations between objects (Relation.has).
    """
    _inheritance_edges: ClassVar[List[Tuple[Type[TrackedObjectMixin], ...]]] = []
    """
    The edges that represent inheritance relations between objects (Relation.isA).
    """

    @classmethod
    def _reset_dependency_graph(cls) -> None:
        """
        Reset the dependency graph and all class indices.
        """
        cls._dependency_graph = rx.PyDAG()
        cls._class_graph_indices = {}
        cls._composition_edges = []
        cls._inheritance_edges = []

    @classmethod
    @final
    @lru_cache(maxsize=None)
    def has(cls, tracked_object_type: Type[TrackedObjectMixin], recursive: bool = False) -> bool:
        neighbors = cls._dependency_graph.adj_direction(cls._my_graph_idx(), Direction.OUTBOUND.value)
        curr_val = any(e == Relation.has and cls._dependency_graph.get_node_data(n).is_a(tracked_object_type)
                       or e == Relation.isA and cls._dependency_graph.get_node_data(n).has(tracked_object_type)
                       for n, e in neighbors.items())
        if recursive:
            return curr_val or any((e == Relation.has
                                    and cls._dependency_graph.get_node_data(n).has(tracked_object_type, recursive=True))
                                   for n, e in neighbors.items())
        else:
            return curr_val

    @classmethod
    @final
    @lru_cache(maxsize=None)
    def is_a(cls, tracked_object_type: Type[TrackedObjectMixin]) -> bool:
        """
        Check if the class is a subclass of the tracked object type.
        """
        return issubclass(cls, tracked_object_type)

    @classmethod
    @lru_cache(maxsize=None)
    def depends_on(cls, tracked_object_type: Type[TrackedObjectMixin]) -> bool:
        raise NotImplementedError

    @classmethod
    def _my_graph_idx(cls):
        return cls._class_graph_indices[cls]

    @classmethod
    def make_class_dependency_graph(cls, composition: bool = True):
        """
        Create a direct acyclic graph containing the class hierarchy.

        :param composition: If True, the class dependency graph will include composition relations.
        """
        subclasses = recursive_subclasses(TrackedObjectMixin)
        subclasses = [sc for sc in subclasses if sc.__name__ != TrackedObjectMixin.__name__]
        for clazz in subclasses:
            cls._add_class_to_dependency_graph(clazz)

            bases = [base for base in clazz.__bases__ if
                     base.__module__ not in ["builtins"] and base in subclasses]

            for base in bases:
                cls._add_class_to_dependency_graph(base)
                if (clazz, base) in cls._inheritance_edges:
                    continue
                cls._dependency_graph.add_edge(cls._class_graph_indices[clazz], cls._class_graph_indices[base],
                                               Relation.isA)
                cls._inheritance_edges.append((clazz, base))

        if not composition:
            return

        for clazz, idx in cls._class_graph_indices.items():
            if clazz.__module__ == "builtins":
                continue
            clazz.parse_fields()

    @classmethod
    @lru_cache(maxsize=None)
    def parse_fields(cls) -> None:

        for f in cls.get_fields():

            logger.debug("=" * 80)
            logger.debug(f"Processing Field {cls.__name__}.{f.name}: {f.type}.")

            # skip private fields
            if f.name.startswith("_"):
                logger.debug(f"Skipping since the field starts with _.")
                continue

            field_info = FieldInfo(cls, f)
            cls.parse_field(field_info)

    @classmethod
    @lru_cache(maxsize=None)
    def get_fields(cls) -> List[Field]:
        skip_fields = []
        bases = [base for base in cls.__bases__ if issubclass(base, TrackedObjectMixin)]
        for base in bases:
            skip_fields.extend(base.get_fields())

        result = [cls_field for cls_field in fields(cls) if cls_field not in skip_fields]

        return result

    @classmethod
    def parse_field(cls, field_info: FieldInfo):
        parent_idx = cls._class_graph_indices[field_info.clazz]
        field_cls: Optional[Type[TrackedObjectMixin]] = None
        field_relation = Relation.has
        if len(field_info.type) == 1 and issubclass(field_info.type[0], TrackedObjectMixin):
            field_cls = field_info.type[0]
        else:
            # TODO: Create a new TrackedObjectMixin class for new type
            logger.debug(f"Skipping unhandled field type: {field_info.type}")
            # logger.debug(f"Creating new TrackedObject type for builtin type {field_info.type}.")
            # field_cls = cls._create_tracked_object_class_for_field(field_info)

        if field_cls is not None:
            if (field_info.clazz, field_cls) in cls._composition_edges:
                return
            cls._add_class_to_dependency_graph(field_cls)
            cls._dependency_graph.add_edge(parent_idx, cls._class_graph_indices[field_cls], field_relation)
            cls._composition_edges.append((field_info.clazz, field_cls))

    @classmethod
    def _create_tracked_object_class_for_field(cls, field_info: FieldInfo):
        raise NotImplementedError

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
        # if name not in [f.name for f in fields(TrackedObjectMixin)] + ['has', 'is_a', 'depends_on']\
        #         and not name.startswith("_"):
        #     self._record_dependency(name)
        return object.__getattribute__(self, name)

    def _record_dependency(self, attr_name):
        # Inspect stack to find instance of CallableExpression
        for frame_info in inspect.stack():
            func_name = frame_info.function
            local_self = frame_info.frame.f_locals.get("self", None)
            if (
                    func_name == "__call__" and
                    local_self is not None and
                    type(local_self).__module__ == "callable_expression" and
                    type(local_self).__name__ == "CallableExpression"
            ):
                logger.debug("TrackedObject used inside CallableExpression")
                break


annotations = TrackedObjectMixin.__annotations__
for val in [f.name for f in fields(TrackedObjectMixin)]:
    annotations.pop(val, None)
