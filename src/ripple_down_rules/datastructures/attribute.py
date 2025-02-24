from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Union, Dict, Type, Self, Any, Set, List, Optional

from ripple_down_rules.datastructures.dataclasses import Range
from ripple_down_rules.datastructures.enums import CategoryValueType, CategoricalValue
from ripple_down_rules.utils import make_set, make_value_or_raise_error


class Attribute(ABC):
    """
    An attribute is a name-value pair that represents a feature of a case.
    an attribute can be used to compare two cases, to make a conclusion (which is also an attribute) about a case.
    """
    mutually_exclusive: bool = False
    """
    Whether the attribute is mutually exclusive, this means that the attribute instance can only have one value.
    """
    value_type: CategoryValueType = CategoryValueType.Nominal
    """
    The type of the value of the attribute.
    """
    _range: Union[set, Range] = None
    """
    The range of the attribute, this can be a set of possible values or a range of numeric values (int, float).
    """

    _registry: Dict[str, Type[Self]] = {}
    """
    A dictionary of all dynamically created subclasses of the attribute class.
    """

    @classmethod
    def create_attribute(cls, name: str, mutually_exclusive: bool, value_type: CategoryValueType,
                         range_: Union[set, Range], **kwargs) \
            -> Type[Self]:
        """
        Create a new attribute subclass.

        :param name: The name of the attribute.
        :param mutually_exclusive: Whether the attribute is mutually exclusive.
        :param value_type: The type of the value of the attribute.
        :param range_: The range of the attribute.
        :return: The new attribute subclass.
        """
        kwargs.update(mutually_exclusive=mutually_exclusive, value_type=value_type, _range=range_)
        if name in cls._registry:
            if not cls._registry[name].mutually_exclusive == mutually_exclusive:
                print(f"Mutually exclusive of {name} is different from {cls._registry[name].mutually_exclusive}.")
                cls._registry[name].mutually_exclusive = mutually_exclusive
            if not cls._registry[name].value_type == value_type:
                raise ValueError(f"Value type of {name} is different from {cls._registry[name].value_type}.")
            if not cls._registry[name].within_range(range_):
                if isinstance(cls._registry[name]._range, set):
                    cls._registry[name]._range.union(range_)
                else:
                    raise ValueError(f"Range of {name} is different from {cls._registry[name]._range}.")
            return cls._registry[name]
        new_attribute_type: Type[Self] = type(name.lower(), (cls,), {}, **kwargs)
        cls.register(new_attribute_type)
        return new_attribute_type

    @classmethod
    def register(cls, subclass: Type[Attribute]):
        """
        Register a subclass of the attribute class, this is used to be able to dynamically create Attribute subclasses.

        :param subclass: The subclass to register.
        """
        if not issubclass(subclass, Attribute):
            raise ValueError(f"{subclass} is not a subclass of Attribute.")
        # Add the subclass to the registry if it is not already in the registry.
        if subclass not in cls._registry:
            cls._registry[subclass.__name__.lower()] = subclass
        else:
            raise ValueError(f"{subclass} is already registered.")

    @classmethod
    def get_subclass(cls, name: str) -> Type[Attribute]:
        """
        Get a subclass of the attribute class by name.

        :param name: The name of the subclass.
        :return: The subclass.
        """
        if ' ' in name or '_' in name:
            name = name.capitalize().strip(' _')

        if name.lower() in cls._registry:
            return cls._registry[name.lower()]
        raise ValueError(f"No subclass with name {name}.")

    def __init_subclass__(cls, **kwargs):
        """
        Set the name of the attribute class to the name of the class in lowercase.
        """
        super().__init_subclass__()
        cls.name = cls.__name__.lower()

        mutually_exclusive = kwargs.get("mutually_exclusive", None)
        value_type = kwargs.get("value_type", None)
        _range = kwargs.get("_range", None)

        cls.mutually_exclusive = mutually_exclusive if mutually_exclusive else cls.mutually_exclusive
        cls.value_type = value_type if value_type else cls.value_type
        cls._range = _range if _range else cls._range

    def __init__(self, value: Any):
        """
        Create an attribute.

        :param value: The value of the attribute.
        """
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: Any):
        value = self.make_value(value)
        if not self.mutually_exclusive:
            self._value = make_set(value)
        else:
            self._value = value

    @classmethod
    def make_value(cls, value: Any) -> Any:
        """
        Make a value for the attribute.

        :param value: The value to make.
        """
        if not cls.is_possible_value(value):
            raise ValueError(f"Value {value} is not a possible value for {cls.__name__} with range {cls._range}.")
        if cls.value_type == CategoryValueType.Iterable:
            if not hasattr(value, "__iter__") or isinstance(value, str):
                value = [value]
        elif not cls.mutually_exclusive:
            value = make_set(value)
        else:
            value = make_value_or_raise_error(value)
        return cls._make_value(value)

    @classmethod
    @abstractmethod
    def _make_value(cls, value: Any) -> Any:
        """
        Make a value for the attribute.

        :param value: The value to make.
        """
        pass

    def __eq__(self, other: Attribute):
        if not isinstance(other, Attribute):
            return False
        if self.name != other.name:
            return False
        if isinstance(self.value, set) and not isinstance(other.value, set):
            return self.value == make_set(other.value)
        else:
            return self.value == other.value

    @classmethod
    @abstractmethod
    def is_possible_value(cls, value: Any) -> bool:
        """
        Check if a value is a possible value for the attribute or if it can be converted to a possible value.

        :param value: The value to check.
        """
        pass

    @classmethod
    def is_within_range(cls, value: Union[set, Range, Any]) \
            -> bool:
        """
        Check if a value is within the range of the attribute.

        :param value: The value to check.
        :return: Boolean indicating whether the value is within the range or not.
        """
        if isinstance(cls._range, set):
            if hasattr(value, "__iter__") and not isinstance(value, str):
                return set(value).issubset(cls._range)
            elif isinstance(value, Range):
                return False
            elif isinstance(value, str):
                return value.lower() in cls._range
            else:
                return value in cls._range
        else:
            return value in cls._range

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f"{self.name}: {self.value}"

    def __repr__(self):
        return self.__str__()


class Integer(Attribute):
    """
    A discrete attribute is an attribute that has a value that is a discrete category.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Ordinal
    _range: Range = Range(-float("inf"), float("inf"), min_closed=False, max_closed=False)

    @classmethod
    def _make_value(cls, value: Any) -> int:
        return int(value)

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.isdigit() and cls.is_within_range(int(value))
        else:
            return isinstance(value, int) and cls.is_within_range(value)


class Continuous(Attribute):
    """
    A continuous attribute is an attribute that has a value that is a continuous category.
    """
    mutually_exclusive: bool = False
    value_type = CategoryValueType.Continuous
    _range: Range = Range(-float("inf"), float("inf"), min_closed=False, max_closed=False)

    @classmethod
    def _make_value(cls, value: Any) -> float:
        return float(value)

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.replace(".", "", 1).isdigit() and cls.is_within_range(float(value))
        return isinstance(value, (float, int)) and cls.is_within_range(value)


class Categorical(Attribute, ABC):
    """
    A categorical attribute is an attribute that has a value that is a category.
    """
    mutually_exclusive: bool = False
    value_type = CategoryValueType.Nominal
    _range: Set[Union[str, type]] = None
    Values: Type[CategoricalValue]

    def __init_subclass__(cls, **kwargs):
        """
        Create the Values enum class for the categorical attribute, this enum class contains all the possible values
        of the attribute.
        Note: This method is called when a subclass of Categorical is created (not when an instance is created).
        """
        super().__init_subclass__(**kwargs)
        if not cls._range:
            cls._range = set()
        cls.create_values()

    def __init__(self, value: Union[Categorical.Values, str]):
        super().__init__(value)

    @classmethod
    def _make_value(cls, value: Union[str, Categorical.Values, Set[str]]) -> Union[Set, Categorical.Values]:
        if len(cls._range) == 0:
            if isinstance(value, str):
                cls.add_new_category(value)
            else:
                cls.add_new_category(type(value))
            return cls._make_value(value)
        if isinstance(value, str) and len(cls._range) > 0 and type(list(cls._range)[0]) == str:
            return cls.Values[value.lower()]
        elif isinstance(value, cls.Values) or any(
                isinstance(v, type) and isinstance(value, v) for v in cls.Values.to_list()):
            return value
        elif isinstance(value, set):
            return {cls._make_value(v) for v in value}
        else:
            raise ValueError(f"Value {value} should be a string or a CategoricalValue.")

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if len(cls._range) == 0:
            if isinstance(value, str):
                cls.add_new_category(value)
            else:
                cls.add_new_category(type(value))
            return cls.is_possible_value(value)
        if len(cls._range) > 0 and type(list(cls._range)[0]) == str:
            return cls.is_within_range(value)
        elif isinstance(value, cls.Values) or any(isinstance(v, type) and isinstance(value, v) for v in cls.Values.to_list()):
            return True
        else:
            return False

    @classmethod
    def from_str(cls, category: str):
        return cls(cls.Values[category.lower()])

    @classmethod
    def from_strs(cls, categories: List[str]):
        return [cls.from_str(c) for c in categories]

    @classmethod
    def add_new_categories(cls, categories: List[str]):
        for category in categories:
            cls.add_new_category(category)

    @classmethod
    def add_new_category(cls, category: str):
        if isinstance(category, str):
            cls._range.add(category.lower())
        elif isinstance(category, type):
            cls._range.add(category)
        else:
            raise ValueError(f"Category {category} should be a string or a type.")
        cls.create_values()

    @classmethod
    def create_values(cls):
        if all(isinstance(c, str) for c in cls._range):
            cls.Values = CategoricalValue(f"{cls.__name__}Values", {c.lower(): c.lower() for c in cls._range})
        else:
            cls.Values = CategoricalValue(f"{cls.__name__}Values",
                                          {c.__name__.lower(): c for c in cls._range})


class ListOf(Attribute, ABC):
    """
    A list of attribute is an attribute that has a value that is a list of other attributes, but all the attributes in
    the list must have the same type.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Iterable
    _range: Set[Attribute]
    element_type: Type[Attribute]
    list_size: Optional[int] = None

    def __init__(self, value: List[element_type]):
        super().__init__(value)

    @classmethod
    def create_attribute(cls, name: str, element_type: Type[Attribute],
                         list_size: Optional[int] = None) -> Type[Self]:
        """
        Create a new attribute subclass that is a list of other attributes ot type _element_type.

        :param name: The name of the attribute.
        :param element_type: The type of the elements in the list.
        :param list_size: The size of the list.
        :return: The new attribute subclass.
        """
        return super().create_attribute(name, cls.mutually_exclusive, cls.value_type, make_set(element_type),
                                        element_type=element_type, list_size=list_size)

    def __init_subclass__(cls, **kwargs):
        """
        """
        super().__init_subclass__(**kwargs)
        element_type = kwargs.get("element_type", None)
        list_size = kwargs.get("list_size", None)
        if not element_type:
            raise ValueError("ListOf subclasses must have an element_type.")
        cls.element_type = element_type
        cls.list_size = list_size

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if not hasattr(value, "__iter__") or isinstance(value, str):
            value = [value]
        if cls.list_size and len(value) != cls.list_size:
            return False
        else:
            return all(cls.element_type.is_possible_value(v) for v in value)

    @classmethod
    def _make_value(cls, value: Any) -> List[element_type]:
        if cls.list_size and len(value) != cls.list_size:
            raise ValueError(f"Value {value} should be a list with size {cls.list_size},"
                             f"got list of size {len(value)} instead.")
        value = [cls.element_type.make_value(v) if not isinstance(v, cls.element_type) else v for v in value]
        return value


class DictOf(Attribute, ABC):
    """
    A dictionary of attribute is an attribute that has a value that is a dictionary of other attributes, but all the
    attributes in the dictionary must have the same type.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Iterable
    _range: Set[Attribute]
    element_type: Type[Attribute]

    def __init__(self, value: Dict[str, element_type]):
        super().__init__(value)

    @classmethod
    def create_attribute(cls, name: str, element_type: Type[Attribute]) -> Type[Self]:
        """
        Create a new attribute subclass that is a dictionary of other attributes ot type _element_type.

        :param name: The name of the attribute.
        :param element_type: The type of the elements in the dictionary.
        :return: The new attribute subclass.
        """
        return super().create_attribute(name, cls.mutually_exclusive, cls.value_type, make_set(element_type),
                                        element_type=element_type)

    def __init_subclass__(cls, **kwargs):
        """
        """
        super().__init_subclass__(**kwargs)
        element_type = kwargs.get("element_type", None)
        if not element_type:
            raise ValueError("DictOf subclasses must have an element_type.")
        cls.element_type = element_type

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        else:
            return all(cls.element_type.is_possible_value(v) for v in value.values())

    @classmethod
    def _make_value(cls, value: Any) -> Dict[str, element_type]:
        value = {k: cls.element_type.make_value(v) if not isinstance(v, cls.element_type) else v for k, v in
                 value.items()}
        return value


class Bool(Attribute):
    """
    A binary attribute is an attribute that has a value that is a binary category.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Binary
    _range: set = {True, False}

    def __init__(self, value: Union[bool, str, int, float]):
        super().__init__(value)

    @classmethod
    def _make_value(cls, value: Any) -> bool:
        if isinstance(value, str):
            if value.lower() in ["true", "1", "1.0"]:
                return True
            elif value.lower() in ["false", "0", "0.0"]:
                return False
        else:
            return bool(value)

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.lower().strip() in ["true", "false", "1", "0", "1.0", "0.0"]
        if isinstance(value, bool):
            return True
        if isinstance(value, int):
            return value in [0, 1]
        if isinstance(value, float):
            return value in [0.0, 1.0]
        return False


class Unary(Attribute):
    """
    A unary attribute is an attribute that has a value that is a unary category.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Unary
    _range: set

    def __init__(self):
        super().__init__(self.__class__.__name__)

    @classmethod
    def _make_value(cls, value: Any) -> str:
        return cls.__name__

    @classmethod
    def is_possible_value(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.lower() == cls.__name__.lower()
        return False


class Stop(Unary):
    """
    A stop category is a special category that represents the stopping of the classification to prevent a wrong
    conclusion from being made.
    """


class Species(Categorical):
    """
    A species category is a category that represents the species of an animal.
    """
    mutually_exclusive: bool = True
    value_type = CategoryValueType.Nominal
    _range: set = {"mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"}

    def __init__(self, species: Union[Species.Values, str]):
        super().__init__(species)


class Habitat(Categorical):
    """
    A habitat category is a category that represents the habitat of an animal.
    """
    mutually_exclusive: bool = False
    value_type = CategoryValueType.Nominal
    _range: set = {"land", "water", "air"}

    def __init__(self, habitat: Union[Habitat.Values, str]):
        super().__init__(habitat)
