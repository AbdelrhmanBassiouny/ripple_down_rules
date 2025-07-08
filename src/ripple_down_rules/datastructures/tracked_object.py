from __future__ import annotations
import inspect
import uuid
from dataclasses import dataclass, field
import rustworkx as rx
from typing_extensions import Any, TYPE_CHECKING, Type, final, ClassVar

from .callable_expression import CallableExpression

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
    def depends_on(cls, tracked_object_type: Type[TrackedObjectMixin]) -> bool:
        ...

    def __getattribute__(self, name: str) -> Any:
        if name not in ['_rdr_rule', '_rdr', '_rdr_tracked_object_id', '_dependency_graph']:
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
