from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Union, Dict, Type, Self, Any, Set, List, Optional, Tuple, get_type_hints, get_origin, \
    get_args

from ripple_down_rules.datastructures.dataclasses import Range
from ripple_down_rules.datastructures.enums import ValueType, CategoricalValue
from ripple_down_rules.utils import make_set, make_value_or_raise_error, can_be_a_set


class AbstractAttribute(ABC):
    _mutually_exclusive: bool
    """
    Whether the attribute is mutually exclusive, this means that the attribute instance can only have one value.
    """
    _value_type: ValueType
    """
    The type of the value of the attribute.
    """
    _value_range: Union[set, Range]
    """
    The range of the attribute, this can be a set of possible values or a range of numeric values (int, float).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the attribute.
        """
        pass

    @property
    @abstractmethod
    def value(self) -> Any:
        """
        The value of the attribute.
        """
        pass

    @value.setter
    @abstractmethod
    def value(self, value: Any):
        """
        Set the value of the attribute.
        """
        pass

    @property
    def mutually_exclusive(self) -> bool:
        return self._mutually_exclusive

    @property
    def value_type(self) -> ValueType:
        return self._value_type

    @property
    def value_range(self) -> Union[set, Range]:
        return self._value_range

    def __getitem__(self, item: str) -> AbstractAttribute:
        return self._attributes[item.lower()]

    def __setitem__(self, key: str, value: AbstractAttribute):
        self._attributes[key.lower()] = value


class Attribute(AbstractAttribute):
    """
    An attribute is a name-value pair that represents a feature of a case.
    an attribute can be used to compare two cases, to make a conclusion (which is also an attribute) about a case.
    """
    mutually_exclusive: bool = False
    """
    Whether the attribute is mutually exclusive, this means that the attribute instance can only have one value.
    """
    value_type: ValueType = ValueType.Nominal
    """
    The type of the value of the attribute.
    """
    value_range: Union[set, Range] = None
    """
    The range of the attribute, this can be a set of possible values or a range of numeric values (int, float).
    """
    registry: Dict[str, Type[Attribute]] = {}
    """
    A dictionary of all dynamically created subclasses of the attribute class.
    """

    @classmethod
    def create_attribute(cls, name: str, mutually_exclusive: bool, value_type: ValueType,
                         range_: Union[set, Range], **kwargs) \
            -> Type[Attribute]:
        """
        Create a new attribute subclass.

        :param name: The name of the attribute.
        :param mutually_exclusive: Whether the attribute is mutually exclusive.
        :param value_type: The type of the value of the attribute.
        :param range_: The range of the attribute.
        :return: The new attribute subclass.
        """
        kwargs.update(mutually_exclusive=mutually_exclusive, value_type=value_type, value_range=range_)
        if name in cls.registry:
            if not cls.registry[name].mutually_exclusive == mutually_exclusive:
                print(f"Mutually exclusive of {name} is different from {cls.registry[name].mutually_exclusive}.")
                cls.registry[name].mutually_exclusive = mutually_exclusive
            if not cls.registry[name].value_type == value_type:
                raise ValueError(f"Value type of {name} is different from {cls.registry[name].value_type}.")
            if not cls.registry[name].is_within_range(range_):
                if isinstance(cls.registry[name].value_range, set):
                    cls.registry[name].value_range.update(range_)
                else:
                    raise ValueError(f"Range of {name} is different from {cls.registry[name].value_range}.")
            return cls.registry[name]
        new_attribute_type: Type[Self] = type(name.lower(), (cls,), {}, **kwargs)
        cls.register(new_attribute_type)
        return new_attribute_type

    def __len__(self):
        if hasattr(self.value, "__len__"):
            return len(self.value)
        else:
            return int(self.value is not None)

    def __contains__(self, item):
        if hasattr(self.value, "__contains__"):
            return item in self.value
        else:
            return self.value == item

    @classmethod
    def register(cls, subclass: Type[Attribute]):
        """
        Register a subclass of the attribute class, this is used to be able to dynamically create Attribute subclasses.

        :param subclass: The subclass to register.
        """
        if not issubclass(subclass, Attribute):
            raise ValueError(f"{subclass} is not a subclass of Attribute.")
        # Add the subclass to the registry if it is not already in the registry.
        if subclass not in cls.registry:
            cls.registry[subclass.__name__.lower()] = subclass
        else:
            raise ValueError(f"{subclass} is already registered.")

    def __init_subclass__(cls, **kwargs):
        """
        Set the name of the attribute class to the name of the class in lowercase.
        """
        super().__init_subclass__()

        mutually_exclusive = kwargs.get("mutually_exclusive", None)
        value_type = kwargs.get("value_type", None)
        value_range = kwargs.get("value_range", None)

        cls.mutually_exclusive = mutually_exclusive if mutually_exclusive else cls.mutually_exclusive
        cls.value_type = value_type if value_type else cls.value_type
        cls.value_range = value_range if value_range else cls.value_range

    def __init__(self, value: Any):
        """
        Create an attribute.

        :param value: The value of the attribute.
        """
        self.value = value

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def as_dict(self):
        return {self.name: self.value}

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: Any):
        if not self.mutually_exclusive:
            value = make_set(value)
        else:
            value = value
        self._value = self.make_value(value)

    @classmethod
    def make_value(cls, value: Any) -> Any:
        """
        Make a value for the attribute.

        :param value: The value to make.
        """
        if not cls.is_possible_value(value):
            raise ValueError(f"Value {value} is not a possible value for {cls.__name__} with range {cls.value_range}.")
        if cls.value_type == ValueType.Iterable:
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
        if isinstance(cls.value_range, set):
            if hasattr(value, "__iter__") and not isinstance(value, str):
                return set(value).issubset(cls.value_range)
            elif isinstance(value, Range):
                return False
            elif isinstance(value, str):
                return value.lower() in cls.value_range
            else:
                return value in cls.value_range
        else:
            return value in cls.value_range

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
    value_type = ValueType.Ordinal
    value_range: Range = Range(-float("inf"), float("inf"), min_closed=False, max_closed=False)

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
    value_type = ValueType.Continuous
    value_range: Range = Range(-float("inf"), float("inf"), min_closed=False, max_closed=False)

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
    value_type = ValueType.Nominal
    value_range: Set[Union[str, type]] = None
    Values: Type[CategoricalValue]

    def __init_subclass__(cls, **kwargs):
        """
        Create the Values enum class for the categorical attribute, this enum class contains all the possible values
        of the attribute.
        Note: This method is called when a subclass of Categorical is created (not when an instance is created).
        """
        super().__init_subclass__(**kwargs)
        if not cls.value_range:
            cls.value_range = set()
        cls.create_values()

    def __init__(self, value: Union[Categorical.Values, str]):
        super().__init__(value)

    @classmethod
    def _make_value(cls, value: Union[str, Categorical.Values, Set[str]]) -> Union[Set, Categorical.Values]:
        if isinstance(value, str) and len(cls.value_range) > 0 and type(list(cls.value_range)[0]) == str:
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
        if len(cls.value_range) == 0:
            raise ValueError(f"Attribute {cls.__name__} has no possible values.")
        elif len(cls.value_range) > 0 and type(list(cls.value_range)[0]) == str:
            return cls.is_within_range(value)
        elif isinstance(value, cls.Values) or any(
                isinstance(v, type) and isinstance(value, v) for v in cls.Values.to_list()):
            return True
        elif isinstance(value, set):
            return all(cls.is_possible_value(v) for v in value)
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
            cls.value_range.add(category.lower())
        elif isinstance(category, type):
            cls.value_range.add(category)
        else:
            raise ValueError(f"Category {category} should be a string or a type.")
        cls.create_values()

    @classmethod
    def create_values(cls):
        if len(cls.value_range) > 0 and all(isinstance(c, str) for c in cls.value_range):
            cls.Values = CategoricalValue(f"{cls.__name__}Values", {c.lower(): c.lower() for c in cls.value_range})
        else:
            cls.Values = CategoricalValue(f"{cls.__name__}Values",
                                          {c.__name__.lower(): c for c in cls.value_range})


class ListOf(Attribute, ABC):
    """
    A list of attribute is an attribute that has a value that is a list of other attributes, but all the attributes in
    the list must have the same type.
    """
    mutually_exclusive: bool = True
    value_type = ValueType.Iterable
    value_range: Set[Attribute]
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
    value_type = ValueType.Iterable
    value_range: Set[Attribute]
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
    value_type = ValueType.Binary
    value_range: set = {True, False}

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
    value_type = ValueType.Unary
    value_range: set

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
    value_type = ValueType.Nominal
    value_range: set = {"mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"}

    def __init__(self, species: Union[Species.Values, str]):
        super().__init__(species)


class Habitat(Categorical):
    """
    A habitat category is a category that represents the habitat of an animal.
    """
    mutually_exclusive: bool = False
    value_type = ValueType.Nominal
    value_range: set = {"land", "water", "air"}

    def __init__(self, habitat: Union[Habitat.Values, str]):
        super().__init__(habitat)

    @property
    def as_dict(self):
        return {self.name: self.value}


def get_attributes_from_object(obj: Any) -> List[Attribute]:
    """
    Get the attributes of an object.

    :param obj: The object to get the attributes from.
    :return: The attributes of the object.
    """
    attributes = []
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        if attr_name.startswith("_") or callable(attr):
            continue
        matched_attribute = get_or_create_matching_attribute(attr_name, attr, obj)
        attributes.append(matched_attribute)
    return attributes


def get_or_create_matching_attribute(attr_name: str, attr_value: Optional[Any] = None, obj: Optional[Any] = None,
                                     type_hint: Optional[Type] = None) -> Union[Attribute, Type[Attribute]]:
    """
    Get or create a matching attribute type for an attribute value.

    :param attr_name: The name of the attribute.
    :param obj: The object to get the attributes from.
    :param attr_value: The value of the attribute.
    :param type_hint: The value type hint to match the attribute value with.
    :return: The matching attribute type instantiated with the attribute value.
    """
    if not type_hint and attr_value and obj:
        return get_or_create_attribute_from_value(attr_name, attr_value, obj)
    if not type_hint and obj:
        type_hint, origin, args = get_hint_for_attribute(attr_name, obj)
    if not type_hint:
        raise ValueError(f"Couldn't get type for Attribute {attr_name}, please provide a type hint")
    origin, args = get_origin(type_hint), get_args(type_hint)
    if origin == Union and len(args) == 2 and args[1] == type(None):
        origin = args[0]
    if origin in [list, tuple, dict]:
        if args[0] in [int, float, str, bool] or origin == dict and args[1] in [int, float, str, bool]:
            attr_type = _get_or_create_attribute_from_iterable(attr_name, value_hint=type_hint)
        else:
            attr_type = Categorical.create_attribute(attr_name, False,
                                                     ValueType.Nominal, make_set(args))
    elif len(args) == 1 and args[0] == int:
        attr_type = Integer
    elif len(args) == 1 and args[0] == float:
        attr_type = Continuous
    elif len(args) == 1 and args[0] == bool:
        attr_type = Bool
    elif len(args) == 1 and args[0] == Any:
        raise ValueError(f"Couldn't get type for Attribute {attr_name}, please provide a type hint")
    else:
        args = args or type_hint
        attr_type = Categorical.create_attribute(attr_name, True, ValueType.Nominal,
                                                 make_set(args))
    return attr_type(attr_value)


def get_or_create_attribute_from_value(attr_name: str, attr_value: Any, obj: Any) -> Attribute:
    """
    Get or create an attribute for an attribute value.

    :param attr_name: The name of the attribute.
    :param attr_value: The value of the attribute.
    :param obj: The object to get the attributes from.
    :return: The attribute type.
    """
    iterable = hasattr(attr_value, "__iter__") and not isinstance(attr_value, str)
    if iterable and not can_be_a_set(attr_value):
        attr_type = _get_or_create_attribute_from_iterable(attr_name, attr_value)
    else:
        if Integer.is_possible_value(attr_value):
            attr_type = Integer
        elif Continuous.is_possible_value(attr_value):
            attr_type = Continuous
        elif Bool.is_possible_value(attr_value):
            attr_type = Bool
        elif iterable:
            attr_type, attr_value = _get_or_create_attribute_from_set(attr_name, attr_value, obj)
        else:
            attr_type = Categorical.create_attribute(attr_name, True, ValueType.Nominal,
                                                     make_set(type(attr_value)))
    return attr_type(attr_value)


def _get_or_create_attribute_from_iterable(attr_name: str, attr_value: Optional[Any] = None,
                                           value_hint: Optional[Type] = None) -> Type[Attribute]:
    """
    Get the attribute type for an iterable attribute.

    :param attr_name: The name of the attribute.
    :param attr_value: The value of the attribute.
    :param value_hint: The value type hint to match the attribute value with.
    :return: The attribute type.
    """
    iterable_type = ListOf
    if attr_value:
        values = attr_value
        if type(attr_value) == dict:
            values = list(attr_value.values())
            iterable_type = DictOf
        if all(Integer.is_possible_value(v) for v in values):
            attr_type = Integer
        elif all(Continuous.is_possible_value(v) for v in values):
            attr_type = Continuous
        elif all(Bool.is_possible_value(v) for v in values):
            attr_type = Bool
        else:
            raise ValueError(f"Attribute {attr_name} is not a valid iterable.")
    elif value_hint:
        origin, args = get_origin(value_hint), get_args(value_hint)
        if origin == Union and len(args) == 2 and args[1] == type(None):
            origin = args[0]
        if origin in [List, Tuple]:
            iterable_type = ListOf
            value_hint = args[0]
        elif origin == Dict:
            iterable_type = DictOf
            value_hint = args[1]
        if not value_hint:
            raise ValueError(f"Couldn't get type for Attribute {attr_name}, please provide a type hint")
        attr_type = get_or_create_matching_attribute(f"{attr_name}_element", type_hint=value_hint)
    else:
        raise ValueError(f"Couldn't get type for Attribute {attr_name}, please provide a type hint")
    return iterable_type.create_attribute(attr_name, attr_type)


def _get_or_create_attribute_from_set(attr_name: str, attr_value: Any, obj: Any) -> Tuple[
    Type[Attribute], Set[Attribute]]:
    """
    Get the attribute type and value for a set attribute.

    :param attr_name: The name of the attribute.
    :param attr_value: The value of the attribute.
    :param obj: The object to get the attributes from.
    :return: The attribute type and value.
    """
    attr_value = make_set(attr_value)
    attr_value_element = list(attr_value)[0] if len(attr_value) > 0 else None
    attr_value_type = type(attr_value_element) if attr_value_element else None
    if not attr_value_type:
        attr_value_type = get_value_type_from_type_hint(attr_name, obj)
        if attr_value_type in [List, Set, Tuple, Dict]:
            attr_value_type = get_or_create_matching_attribute(attr_name, attr_value_element, obj)
    range_ = make_set(attr_value_type) if attr_value_type else None
    attr_type = Categorical.create_attribute(attr_name, False,
                                             ValueType.Nominal, range_)
    return attr_type, attr_value


def get_value_type_from_type_hint(attr_name: str, obj: Any) -> Type:
    """
    Get the value type from the type hint of an object attribute.

    :param attr_name: The name of the attribute.
    :param obj: The object to get the attributes from.
    """
    hint, origin, args = get_hint_for_attribute(attr_name, obj)
    if not origin:
        raise ValueError(f"Couldn't get type for Attribute {attr_name}, please provide a type hint")
    if origin in [list, set, tuple, type, dict]:
        attr_value_type = args[0]
    else:
        raise ValueError(f"Attribute {attr_name} has unsupported type {hint}.")
    return attr_value_type


def get_hint_for_attribute(attr_name: str, obj: Any) -> Tuple[Type, Type, Tuple[Type]]:
    """
    Get the type hint for an attribute of an object.

    :param attr_name: The name of the attribute.
    :param obj: The object to get the attribute from.
    :return: The type hint of the attribute.
    """
    class_attr = getattr(obj.__class__, attr_name)
    if isinstance(class_attr, property):
        if not class_attr.fget:
            raise ValueError(f"Attribute {attr_name} has no getter.")
        hint = get_type_hints(class_attr.fget)['return']
    else:
        hint = get_type_hints(obj.__class__)[attr_name]
    origin = get_origin(hint)
    args = get_args(hint)
    return hint, origin, args
