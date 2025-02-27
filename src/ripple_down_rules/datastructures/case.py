from __future__ import annotations

from collections import UserDict

import pandas as pd
from ordered_set import OrderedSet
from typing_extensions import Union, List, Optional, Any, Type, Tuple, Dict

from ripple_down_rules.datastructures.attribute import Attribute, ListOf, DictOf, Categorical, Integer, Continuous, \
    Bool, Unary
from ripple_down_rules.datastructures.enums import CategoryValueType
from ripple_down_rules.rules import Rule
from ripple_down_rules.utils import make_set, get_property_name, can_be_a_set, get_attribute_values


class Attributes(UserDict):
    """
    A collection of attributes that represents a set of constraints on a case. This is a dictionary where the keys are
    the names of the attributes and the values are the attributes. All are stored in lower case.
    """

    def __getitem__(self, item: Union[str, Attribute]) -> Attribute:
        if isinstance(item, Attribute):
            return self[item._name]
        else:
            return super().__getitem__(item.lower())

    def __setitem__(self, name: str, value: Attribute):
        name = name.lower()
        if name in self:
            if (isinstance(value, Attribute) and not value._mutually_exclusive) or hasattr(value, "__iter__"):
                self[name]._value.update(make_set(value))
            else:
                raise ValueError(f"Attribute {name} already exists in the case and is mutually exclusive.")
        else:
            super().__setitem__(name, value)

    def __contains__(self, item):
        return super().__contains__(item.lower())

    def __delitem__(self, key):
        super().__delitem__(key.lower())

    def __eq__(self, other):
        if not isinstance(other, (Attributes, dict)):
            return False
        elif isinstance(other, dict):
            return super().__eq__(Attributes(other))
        return super().__eq__(other)


class Case:
    """
    A case is a collection of attributes that represents an instance that can be classified by inferring new attributes
    or additional attribute values for the case.
    """

    def __init__(self, _id: str, attributes: List[Attribute],
                 conclusions: Optional[List[Attribute]] = None,
                 targets: Optional[List[Attribute]] = None,
                 obj: Optional[Any] = None):
        """
        Create a case.

        :param _id: The id of the case.
        :param attributes: The attributes of the case.
        :param conclusions: The conclusions that has been made about the case.
        :param targets: The targets of the case.
        :param obj: The object that the case represents.
        """
        self._attributes = Attributes({a._name: a for a in attributes})
        for attribute in attributes:
            setattr(self, attribute._name, attribute)
        self._id = _id
        self._conclusions: Optional[List[Attribute]] = conclusions
        self._targets: Optional[List[Attribute]] = targets
        self._obj: Any = obj

    @classmethod
    def create_cases_from_dataframe(cls, df: pd.DataFrame, ids: List[str]) -> List[Case]:
        """
        Create cases from a pandas dataframe.

        :param df: pandas dataframe
        :param ids: list of ids
        :return: list of cases
        """
        att_names = df.keys().tolist()
        unique_values: Dict[str, List] = {col_name: df[col_name].unique() for col_name in att_names}
        att_types: Dict[str, Type[Attribute]] = {}
        for col_name, values in unique_values.items():
            values = values.tolist()
            if len(values) == 1:
                att_types[col_name] = type(col_name, (Unary,), {})
            elif len(values) == 2 and all(isinstance(val, bool) or (val in [0, 1]) for val in values):
                att_types[col_name] = type(col_name, (Bool,), {})
            elif len(values) >= 2 and all(isinstance(val, str) for val in values):
                att_types[col_name] = type(col_name, (Categorical,), {'_range': set(values)})
                att_types[col_name].create_values()
            elif len(values) >= 2 and all(isinstance(val, int) for val in values):
                att_types[col_name] = type(col_name, (Integer,), {})
            elif len(values) >= 2 and all(isinstance(val, float) for val in values):
                att_types[col_name] = type(col_name, (Continuous,), {})
        all_cases = []
        for _id, row in zip(ids, df.iterrows()):
            all_att = [att_types[att](row[1][att].item()) for att in att_names]
            all_cases.append(cls(_id, all_att))
        return all_cases

    @classmethod
    def from_object(cls, obj: Any, attributes: Optional[List[Attribute]] = None,
                    conclusions: Optional[List[Attribute]] = None,
                    targets: Optional[List[Attribute]] = None) -> Case:
        """
        Create a case from an object.

        :param obj: The object to create the case from.
        :param attributes: The attributes of the case.
        :param conclusions: The conclusions that has been made about the case.
        :param targets: The targets of the case.
        :return: The case.
        """
        if not attributes:
            attributes = cls.get_attributes_from_object(obj)
        return cls(obj.__class__.__name__, attributes, conclusions, targets, obj=obj)

    def get_property_name(self, property_value: Any) -> str:
        """
        Get the property of the object given its value.

        :param property_value: The value of the property.
        :return: The property.
        """
        name = get_property_name(self._obj, property_value)
        assert name in self._attributes, f"Attribute {name} not found in case."
        return name

    def get_property_from_value(self, property_value: Any) -> Type:
        """
        Get the property of the object given its value.

        :param property_value: The value of the property.
        :return: The property.
        """
        return self.get_property_from_name(get_property_name(self._obj, property_value))

    def __getattr__(self, name):
        """Custom getattr logic."""
        if name.startswith("_") and not name.startswith("__"):
            return object.__getattribute__(self, name)  # Get from self
        return getattr(self._obj, name)  # Get from wrapped object

    def get_property_from_name(self, property_name: str) -> Type:
        """
        Get the property of the object given its name.

        :param property_name: The name of the property.
        :return: The property.
        """
        return getattr(self._obj, property_name)

    @staticmethod
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
            matched_attribute = Case.get_or_create_matching_attribute(attr, attr_name)
            attributes.append(matched_attribute)
        return attributes

    @staticmethod
    def get_or_create_matching_attribute(attr_value: Any, attr_name: str) -> Attribute:
        """
        Get or create a matching attribute type for an attribute value.

        :param attr_value: The value of the attribute.
        :param attr_name: The name of the attribute.
        :return: The matching attribute type instantiated with the attribute value.
        """
        iterable = hasattr(attr_value, "__iter__") and not isinstance(attr_value, str)
        element_attr_name = f"{attr_name}_element"
        if iterable and not can_be_a_set(attr_value):
            iterable_type = ListOf
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
                if list(set(map(type, values))) == [type(values[0])]:
                    attr_type = type(Case.get_or_create_matching_attribute(values[0], element_attr_name))
                else:
                    raise ValueError(f"Cannot create attribute: Values in iterable {attr_name} are of different types, "
                                     f"{list(set(map(type, values)))}.")
            attr_type = iterable_type.create_attribute(attr_name, attr_type)
        else:
            if Integer.is_possible_value(attr_value):
                attr_type = Integer
            elif Continuous.is_possible_value(attr_value):
                attr_type = Continuous
            elif Bool.is_possible_value(attr_value):
                attr_type = Bool
            elif iterable:
                attr_value = make_set(attr_value)
                attr_value_element = list(attr_value)[0] if len(attr_value) > 0 else None
                attr_value_type = type(attr_value_element) if attr_value_element else None
                range_ = make_set(attr_value_type) if attr_value_type else set()
                attr_type = Categorical.create_attribute(attr_name, False,
                                                         CategoryValueType.Nominal, range_)
                if attr_value_element:
                    for sub_attr in dir(attr_value_element):
                        sub_attr_value = set()
                        if sub_attr.startswith("_") or callable(getattr(attr_value_element, sub_attr)):
                            continue
                        for attr_element in attr_value:
                            sub_attr_value.update(get_attribute_values(attr_element, sub_attr))
                        setattr(attr_type, sub_attr, sub_attr_value)
            else:
                attr_type = Categorical.create_attribute(attr_name, False, CategoryValueType.Nominal,
                                                         make_set(type(attr_value)))
        return attr_type(attr_value)

    def remove_attribute(self, attribute_name: str):
        if attribute_name in self:
            del self._attributes[attribute_name]

    def add_attributes(self, attributes: List[Attribute]):
        if not attributes:
            return
        attributes = attributes if isinstance(attributes, list) else [attributes]
        for attribute in attributes:
            self.add_attribute(attribute)

    def add_attribute(self, attribute: Attribute):
        self[attribute._name] = attribute

    def __setitem__(self, attribute_name: str, attribute: Attribute):
        self._attributes[attribute_name] = attribute
        if self._obj:
            setattr(self._obj, attribute_name, attribute)

    @property
    def _attribute_values(self):
        return [a._value for a in self._attributes.values()]

    @property
    def _attributes_list(self):
        return list(self._attributes.values())

    def __eq__(self, other):
        return self._attributes == other._attributes

    def __getitem__(self, attribute_description: Union[str, Attribute, Any]) -> Attribute:
        if isinstance(attribute_description, (Attribute, str)):
            return self._attributes.get(attribute_description, None)
        else:
            return self._attributes[get_property_name(self._obj, attribute_description)]

    def __sub__(self, other):
        return {k: self._attributes[k] for k in self._attributes
                if self._attributes[k] != other._attributes[k]}

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self._attributes
        elif isinstance(item, type) and issubclass(item, Attribute):
            return item.__name__ in self._attributes
        elif isinstance(item, Attribute):
            return item._name in self._attributes and self._attributes[item._name] == item

    @staticmethod
    def _ljust(s, sz=15):
        return str(s).ljust(sz)

    def print_all_names(self, all_names: List[str], max_len: int,
                        target_types: Optional[List[Type[Attribute]]] = None,
                        conclusion_types: Optional[List[Type[Attribute]]] = None) -> int:
        """
        Print all attribute names.

        :param all_names: list of names.
        :param max_len: maximum length.
        :param target_types: list of target types.
        :param conclusion_types: list of category types.
        :return: maximum length.
        """
        all_names_str, max_len = self.get_all_names_str(all_names, max_len, target_types, conclusion_types)
        print(all_names_str)
        return max_len

    def print_values(self, all_names: Optional[List[str]] = None,
                     targets: Optional[List[Attribute]] = None,
                     is_corner_case: bool = False,
                     ljust_sz: int = 15,
                     conclusions: Optional[List[Attribute]] = None):
        print(self.get_values_str(all_names, targets, is_corner_case, conclusions, ljust_sz))

    def __str__(self):
        names, ljust = self.get_all_names_and_max_len()
        row1, ljust = self.get_all_names_str(names, ljust)
        row2 = self.get_values_str(names, ljust_sz=ljust)
        return "\n".join([row1, row2])

    def get_all_names_str(self, all_names: List[str], max_len: int,
                          target_types: Optional[List[Type[Attribute]]] = None,
                          conclusion_types: Optional[List[Type[Attribute]]] = None) -> Tuple[str, int]:
        """
        Get all attribute names, target names and conclusion names.

        :param all_names: list of names.
        :param max_len: maximum length.
        :param target_types: list of target types.
        :param conclusion_types: list of category types.
        :return: string of names, maximum length.
        """
        if conclusion_types or self._conclusions:
            conclusion_types = conclusion_types or list(map(type, self._conclusions))
        category_names = []
        if conclusion_types:
            category_types = conclusion_types or [Attribute]
            category_names = [category_type.__name__.lower() for category_type in category_types]

        if target_types or self._targets:
            target_types = target_types if target_types else list(map(type, self._targets))
        target_names = []
        if target_types:
            target_names = [f"target_{target_type.__name__.lower()}" for target_type in target_types]

        curr_max_len = max(max_len, max([len(name) for name in all_names + category_names + target_names]) + 2)
        names_row = self._ljust(f"names: ", sz=curr_max_len)
        names_row += self._ljust("ID", sz=curr_max_len)
        names_row += "".join(
            [f"{self._ljust(name, sz=curr_max_len)}" for name in all_names + category_names + target_names])
        return names_row, curr_max_len

    def get_all_names_and_max_len(self, all_attributes: Optional[List[Attribute]] = None) -> Tuple[List[str], int]:
        """
        Get all attribute names and the maximum length of the names and values.

        :param all_attributes: list of attributes
        :return: list of names and the maximum length
        """
        all_attributes = all_attributes if all_attributes else self._attributes_list
        all_names = list(OrderedSet([a._name for a in all_attributes]))
        max_len = max([len(name) for name in all_names])
        max_len = max(max_len, max([len(str(a._value)) for a in all_attributes])) + 4
        return all_names, max_len

    def get_values_str(self, all_names: Optional[List[str]] = None,
                       targets: Optional[List[Attribute]] = None,
                       is_corner_case: bool = False,
                       conclusions: Optional[List[Attribute]] = None,
                       ljust_sz: int = 15) -> str:
        """
        Get the string representation of the values of the case.
        """
        all_names = list(self._attributes.keys()) if not all_names else all_names
        targets = targets if targets else self._targets
        if targets:
            targets = targets if isinstance(targets, list) else [targets]
        case_row = self.get_id_and_attribute_values_str(all_names, is_corner_case, ljust_sz)
        case_row += self.get_conclusions_str(conclusions, ljust_sz)
        case_row += self.get_targets_str(targets, ljust_sz)
        return case_row

    def get_id_and_attribute_values_str(self, all_names: Optional[List[str]] = None,
                                        is_corner_case: bool = False,
                                        ljust_sz: int = 15) -> str:
        """
        Get the string representation of the id and names of the case.

        :param all_names: The names of the attributes to include in the string.
        :param is_corner_case: Whether the case is a corner case.
        :param ljust_sz: The size of the ljust.
        """
        all_names = list(self._attributes.keys()) if not all_names else all_names
        if is_corner_case:
            case_row = self._ljust(f"corner case: ", sz=ljust_sz)
        else:
            case_row = self._ljust(f"case: ", sz=ljust_sz)
        case_row += self._ljust(self._id, sz=ljust_sz)
        case_row += "".join([f"{self._ljust(self[name]._value if name in self._attributes else '', sz=ljust_sz)}"
                             for name in all_names])
        return case_row

    def get_targets_str(self, targets: Optional[List[Attribute]] = None, ljust_sz: int = 15) -> str:
        """
        Get the string representation of the targets of the case.
        """
        targets = targets if targets else self._targets
        return self._get_categories_str(targets, ljust_sz)

    def get_conclusions_str(self, conclusions: Optional[List[Attribute]] = None, ljust_sz: int = 15) -> str:
        """
        Get the string representation of the conclusions of the case.
        """
        conclusions = conclusions if conclusions else self._conclusions
        return self._get_categories_str(conclusions, ljust_sz)

    def _get_categories_str(self, categories: List[Union[Attribute, Any]], ljust_sz: int = 15) -> str:
        """
        Get the string representation of the categories of the case.
        """
        if not categories:
            return ""
        categories_str = [self._ljust(c._value if isinstance(c, Attribute) else c, sz=ljust_sz) for c in categories]
        return "".join(categories_str) if len(categories_str) > 1 else categories_str[0]

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        conclusions_cp = self._conclusions.copy() if self._conclusions else None
        return Case(self._id, self._attributes_list.copy(), conclusions_cp)
