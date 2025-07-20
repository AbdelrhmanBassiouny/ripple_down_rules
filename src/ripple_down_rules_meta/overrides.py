from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import ripple_down_rules
from ripple_down_rules import (DependsOn as OGDependsOn, RDRDecorator, TrackedObjectMixin,
                               has, isA)

_overrides = []


def override_ripple_down_rules(name):
    def decorator(obj):
        setattr(getattr(ripple_down_rules, name), '_overridden_by', obj)
        _overrides.append((name, obj))
        return obj

    return decorator


@override_ripple_down_rules("DependsOn")
@dataclass(eq=False)
class DependsOn(OGDependsOn):
    rdr_decorator: RDRDecorator = RDRDecorator(OGDependsOn.models_dir, (bool,), True,
                                                fit=False, package_name="ripple_down_rules_meta")
    """
    An rdr decorator used to fit rules for determining the dependencies.
    """
    @classmethod
    @rdr_decorator.decorator
    def evaluate(cls, dependent: Type[TrackedObjectMixin],
                 dependency: Type[TrackedObjectMixin], recursive: bool = False) -> bool:
        pass

_overrides.append(("dependsOn", DependsOn()))
