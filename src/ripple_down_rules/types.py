from typing_extensions import Union, Iterator, Type, Tuple
from .datastructures.tracked_object import TrackedObjectMixin

PredicateArgElementType = Union[Type[TrackedObjectMixin], TrackedObjectMixin]
PredicateArgType = Union[Iterator[PredicateArgElementType], PredicateArgElementType]
PredicateOutputType = Iterator[Tuple[PredicateArgElementType, PredicateArgElementType]]
