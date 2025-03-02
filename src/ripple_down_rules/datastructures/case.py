from __future__ import annotations

from collections import UserDict

import pandas as pd
from typing_extensions import Union, List, Optional, Any, Type, Dict

from .attribute import Attribute, Categorical, Integer, Continuous, \
    Bool, Unary, get_attributes_from_object
from ..utils import make_set, get_property_name, table_rows_as_str, get_attribute_values_transitively


class Case:
    """
    A case is a collection of attributes that represents an instance that can be classified by inferring new attributes
    or additional attribute values for the case.
    """

    def __init__(self, _id: str, attributes: List[Attribute],
                 conclusions: Optional[List[Attribute]] = None,
                 targets: Optional[List[Attribute]] = None,
                 obj: Optional[Any] = None,
                 is_corner_case: bool = False):
        """
        Create a case.

        :param _id: The id of the case.
        :param attributes: The attributes of the case.
        :param conclusions: The conclusions that has been made about the case.
        :param targets: The targets of the case.
        :param obj: The object that the case represents.
        :param is_corner_case: Whether the case is a corner case (a case that caused the addition of a new rule).
        """
        self.attributes = Attributes({a.name: a for a in attributes})
        self.id = _id
        if conclusions:
            conclusions = conclusions if isinstance(conclusions, list) else [conclusions]
        self.conclusions: Optional[List[Attribute]] = conclusions
        if targets:
            targets = targets if isinstance(targets, list) else [targets]
        self.targets: Optional[List[Attribute]] = targets
        self.obj: Any = obj
        self.is_corner_case: bool = is_corner_case

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
            attributes = get_attributes_from_object(obj)
        return cls(obj.__class__.__name__, attributes, conclusions, targets, obj=obj)

    def get_property_from_value(self, property_value: Any) -> Type:
        """
        Get the property of the object given its value.

        :param property_value: The value of the property.
        :return: The property.
        """
        return self.get_property_from_name(self.get_property_name(property_value))

    def get_property_name(self, property_value: Any) -> str:
        """
        Get the property of the object given its value.

        :param property_value: The value of the property.
        :return: The property.
        """
        name = get_property_name(self.obj, property_value)
        assert name in self.attributes, f"Attribute {name} not found in case."
        return name

    def __getattr__(self, name):
        """Custom getattr logic."""
        if name.startswith("_") and not name.startswith("__"):
            return object.__getattribute__(self, name)  # Get from self
        return getattr(self.obj, name)  # Get from wrapped object

    def get_property_from_name(self, property_name: str) -> Type:
        """
        Get the property of the object given its name.

        :param property_name: The name of the property.
        :return: The property.
        """
        return getattr(self.obj, property_name)

    def add_attributes(self, attributes: List[Attribute]):
        if not attributes:
            return
        attributes = attributes if isinstance(attributes, list) else [attributes]
        for attribute in attributes:
            self.add_attribute(attribute)

    def add_attribute(self, attribute: Attribute):
        self[attribute.name] = attribute

    def __setitem__(self, attribute_name: str, attribute: Attribute):
        self.attributes[attribute_name] = attribute

    @property
    def attribute_values(self):
        return [a.value for a in self.attributes.values()]

    @property
    def attributes_list(self):
        return list(self.attributes.values())

    def __eq__(self, other):
        return self.attributes == other.attributes

    def __getitem__(self, attribute_description: Union[str, Attribute, Any]) -> Attribute:
        if isinstance(attribute_description, (Attribute, str)):
            return self.attributes.get(attribute_description, None)
        else:
            return self.attributes[get_property_name(self.obj, attribute_description)]

    def __sub__(self, other):
        return {k: self.attributes[k] for k in self.attributes
                if self.attributes[k] != other.attributes[k]}

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.attributes
        elif isinstance(item, type) and issubclass(item, Attribute):
            return item.__name__ in self.attributes
        elif isinstance(item, Attribute):
            return item.name in self.attributes and self.attributes[item.name] == item

    def __str__(self):
        attributes: Dict[str, Any] = {"id": self.id}
        attributes.update({a.name: a.value for a in self.attributes_list})
        if self.conclusions:
            attributes.update({c.name: c.value for c in self.conclusions})
        if self.targets:
            attributes.update({t.name: t.value for t in self.targets})
        return table_rows_as_str(attributes)

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        conclusions_cp = self.conclusions.copy() if self.conclusions else None
        targets_cp = self.targets.copy() if self.targets else None
        return Case(self.id, self.attributes_list.copy(), conclusions_cp, targets_cp, self.obj, self.is_corner_case)
