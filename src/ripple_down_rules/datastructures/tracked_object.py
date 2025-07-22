from __future__ import annotations

import inspect
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, Field, fields
from enum import Enum
from functools import lru_cache

import pydot
import rustworkx as rx
from typing_extensions import Any, TYPE_CHECKING, Type, final, ClassVar, Dict, List, Optional, Tuple

from .field_info import FieldInfo
from .. import logger
from ..rules import Rule
from ..utils import recursive_subclasses


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
    _edges: ClassVar[Dict[Relation, List[Tuple[int, int]]]] = defaultdict(list)
    """
    All the edges indexed by relation type.
    """
    _overridden_by: Type[TrackedObjectMixin] = field(init=False, repr=False, hash=False,
                              compare=False, default=None)
    """
    Whether the class has been overridden by a subclass.
    This is used to only include the new class in the dependency graph, not the overridden class.
    """

    @classmethod
    def _reset_dependency_graph(cls) -> None:
        """
        Reset the dependency graph and all class indices.
        """
        cls._dependency_graph = rx.PyDAG()
        cls._class_graph_indices = {}
        cls._edges = defaultdict(list)

    @classmethod
    def _my_graph_idx(cls):
        return cls._class_graph_indices[cls]

    @classmethod
    def make_class_dependency_graph(cls, composition: bool = True):
        """
        Create a direct acyclic graph containing the class hierarchy.

        :param composition: If True, the class dependency graph will include composition relations.
        """
        classes_to_track = recursive_subclasses(cls) + [Rule] + recursive_subclasses(Rule)
        for clazz in classes_to_track:
            cls._add_class_to_dependency_graph(clazz)

        for clazz in cls._class_graph_indices:
            bases = [base for base in clazz.__bases__ if
                     base.__module__ not in ["builtins"] and base in cls._class_graph_indices]

            for base in bases:
                cls._add_class_to_dependency_graph(base)
                clazz_idx = cls._class_graph_indices[clazz]
                base_idx = cls._class_graph_indices[base]
                if (clazz_idx, base_idx) in cls._edges[Relation.isA] or base._overridden_by == clazz:
                    continue
                cls.add_edge(clazz_idx, base_idx, Relation.isA)

        if not composition:
            return
        for clazz, idx in cls._class_graph_indices.items():
            if clazz.__module__ == "builtins":
                continue
            TrackedObjectMixin.parse_fields(clazz)

    @staticmethod
    def parse_fields(clazz) -> None:
        for f in TrackedObjectMixin.get_fields(clazz):

            logger.debug("=" * 80)
            logger.debug(f"Processing Field {clazz.__name__}.{f.name}: {f.type}.")

            # skip private fields
            if f.name.startswith("_"):
                logger.debug(f"Skipping since the field starts with _.")
                continue

            field_info = FieldInfo(clazz, f)
            TrackedObjectMixin.parse_field(field_info)

    @staticmethod
    def get_fields(clazz) -> List[Field]:
        skip_fields = []
        bases = [base for base in clazz.__bases__ if issubclass(base, TrackedObjectMixin)]
        for base in bases:
            skip_fields.extend(TrackedObjectMixin.get_fields(base))

        result = [cls_field for cls_field in fields(clazz) if cls_field not in skip_fields]

        return result

    @staticmethod
    def parse_field(field_info: FieldInfo):
        parent_idx = TrackedObjectMixin._class_graph_indices[field_info.clazz]
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
            field_cls_idx = TrackedObjectMixin._class_graph_indices.get(field_cls, None)
            if field_cls_idx is not None and (parent_idx, field_cls_idx) in TrackedObjectMixin._edges[Relation.has]:
                return
            elif field_cls_idx is None:
                TrackedObjectMixin._add_class_to_dependency_graph(field_cls)
                field_cls_idx = TrackedObjectMixin._class_graph_indices[field_cls]
            TrackedObjectMixin.add_edge(parent_idx, field_cls_idx, field_relation)

    @classmethod
    def add_edge(cls, parent_idx: int, child_idx: int, relation: Relation):
        cls._dependency_graph.add_edge(parent_idx, child_idx, relation)
        cls._edges[relation].append((parent_idx, child_idx))

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
        if class_to_add not in cls._class_graph_indices:
            if not issubclass(class_to_add, TrackedObjectMixin):
                class_to_add._overridden_by = None
            cls_idx = cls._dependency_graph.add_node(class_to_add._overridden_by or class_to_add)
            cls._class_graph_indices[class_to_add] = cls_idx
            if class_to_add._overridden_by:
                cls._class_graph_indices[class_to_add._overridden_by] = cls_idx

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


QueryObject = TrackedObjectMixin
