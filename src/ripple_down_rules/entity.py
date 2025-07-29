from typing import TypeVar, Generic, Type

from typing_extensions import Any, Optional, Union, Iterable

from . import symbolic

T = TypeVar('T')  # Define type variable "T"


def entity(name: str, entity_description: T) -> T:
    return entity_description


def an(entity_type: Type[T], from_: Optional[Any] = None) -> Union[T, Iterable[T]]:
    return symbolic.Variable.from_data(from_, clazz=entity_type)
