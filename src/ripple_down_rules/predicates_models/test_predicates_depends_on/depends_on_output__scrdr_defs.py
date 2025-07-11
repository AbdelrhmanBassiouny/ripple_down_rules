from test.datasets import Drawer, Handle
from typing_extensions import Dict, Optional, Type, Union
from ...rdr_decorators import depends_on
from types import NoneType
from ...datastructures.case import Case
from ...datastructures.tracked_object import TrackedObjectMixin


def conditions_15107258415760040561965086064322061396(case) -> bool:
    def conditions_for_depends_on(parent_type: Type[TrackedObjectMixin], child_type: Type[TrackedObjectMixin], **kwargs) -> bool:
        """Get conditions on whether it's possible to conclude a value for depends_on.output_  of type ."""
        return parent_type.has(child_type)
    return conditions_for_depends_on(**case)


def conclusion_15107258415760040561965086064322061396(case) -> bool:
    def depends_on(parent_type: Type[TrackedObjectMixin], child_type: Type[TrackedObjectMixin], **kwargs) -> bool:
        """Get possible value(s) for depends_on.output_  of type ."""
        return True
    return depends_on(**case)


