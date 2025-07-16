from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from os.path import dirname
from typing import ClassVar

from typing_extensions import Type

from ripple_down_rules import TrackedObjectMixin as OGTrackedObjectMixin
from ripple_down_rules.rdr_decorators import RDRDecorator
from ripple_down_rules.rules import Rule

_overrides = []


def override_ripple_down_rules(name):
    def decorator(obj):
        _overrides.append((name, obj))
        return obj

    return decorator


@override_ripple_down_rules("TrackedObjectMixin")
@dataclass(unsafe_hash=True)
class TrackedObjectMixin(OGTrackedObjectMixin):
    models_dir: ClassVar[str] = os.path.join(dirname(__file__), "predicates_models")
    depends_on_rdr: ClassVar[RDRDecorator] = RDRDecorator(models_dir, (bool,), True,
                                                          package_name="ripple_down_rules", fit=False)

    @classmethod
    @lru_cache(maxsize=None)
    @depends_on_rdr.decorator
    def depends_on(cls, tracked_object_type: Type[TrackedObjectMixin]) -> bool:
        ...

    def __subclasshook__(cls, __subclass):
        if cls is TrackedObjectMixin:
            return True
        return super().__subclasshook__(__subclass)

    @classmethod
    def __subclasses__(cls):
        return super().__subclasses__() + [Rule]


Rule.depends_on = TrackedObjectMixin.depends_on
Rule.has = TrackedObjectMixin.has
assert Rule._dependency_graph is TrackedObjectMixin._dependency_graph
Rule.__subclasshook__ = TrackedObjectMixin.__subclasshook__
