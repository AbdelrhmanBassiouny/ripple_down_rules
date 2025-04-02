from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass
from enum import Enum

from pandas import DataFrame
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase as SQLTable, MappedColumn as SQLColumn, registry
from typing_extensions import Any, Optional, Dict, Type, Set, Hashable, Union, List, TYPE_CHECKING

from ..utils import make_set, row_to_dict, table_rows_as_str, get_value_type_from_type_hint, SubclassJSONSerializer

if TYPE_CHECKING:
    from ripple_down_rules.rules import Rule
    from .callable_expression import CallableExpression


class Case(UserDict, SubclassJSONSerializer):
    """
    A collection of attributes that represents a set of constraints on a case. This is a dictionary where the keys are
    the names of the attributes and the values are the attributes. All are stored in lower case.
    """

    def __init__(self, _id: Optional[Hashable] = None, _type: Optional[Type] = None, **kwargs):
        """
        Create a new row.

        :param _id: The id of the row.
        :param kwargs: The attributes of the row.
        """
        super().__init__(kwargs)
        self._id = _id if _id else id(self)
        self._type = _type

    @classmethod
    def from_obj(cls, obj: Any, obj_name: Optional[str] = None, max_recursion_idx: int = 3) -> Case:
        """
        Create a row from an object.

        :param obj: The object to create a row from.
        :param max_recursion_idx: The maximum recursion index to prevent infinite recursion.
        :param obj_name: The name of the object.
        :return: The row of the object.
        """
        return create_case(obj, max_recursion_idx=max_recursion_idx, obj_name=obj_name)

    def __getitem__(self, item: str) -> Any:
        return super().__getitem__(item.lower())

    def __setitem__(self, name: str, value: Any):
        name = name.lower()
        if name in self:
            if isinstance(self[name], set):
                self[name].update(make_set(value))
            elif isinstance(value, set):
                value.update(make_set(self[name]))
                super().__setitem__(name, value)
            else:
                super().__setitem__(name, make_set([self[name], value]))
        else:
            super().__setitem__(name, value)
        setattr(self, name, self[name])

    def __contains__(self, item):
        if isinstance(item, (type, Enum)):
            item = item.__name__
        return super().__contains__(item.lower())

    def __delitem__(self, key):
        super().__delitem__(key.lower())

    def __hash__(self):
        return self._id

    def _to_json(self) -> Dict[str, Any]:
        serializable = {k: v for k, v in self.items() if not k.startswith("_")}
        serializable["_id"] = self._id
        return {k: v.to_json() if isinstance(v, SubclassJSONSerializer) else v for k, v in serializable.items()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Case:
        id_ = data.pop("_id")
        return cls(_id=id_, **data)


@dataclass
class CaseAttributeValue(SubclassJSONSerializer):
    """
    A column value is a value in a column.
    """
    id: Hashable
    """
    The row id of the column value.
    """
    value: Any
    """
    The value of the column.
    """

    def __eq__(self, other):
        if not isinstance(other, CaseAttributeValue):
            return False
        return self.value == other.value

    def __hash__(self):
        return self.id

    def _to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "value": self.value}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> CaseAttributeValue:
        return cls(id=data["id"], value=data["value"])


class CaseAttribute(set, SubclassJSONSerializer):
    nullable: bool = True
    """
    A boolean indicating whether the case attribute can be None or not.
    """
    mutually_exclusive: bool = False
    """
    A boolean indicating whether the case attribute is mutually exclusive or not. (i.e. can only have one value)
    """

    def __init__(self, values: Set[CaseAttributeValue]):
        """
        Create a new case attribute.

        :param values: The values of the case attribute.
        """
        values = self._type_cast_values_to_set_of_case_attribute_values(values)
        self.id_value_map: Dict[Hashable, Union[CaseAttributeValue, Set[CaseAttributeValue]]] = {id(v): v for v in values}
        super().__init__([v.value for v in values])

    @staticmethod
    def _type_cast_values_to_set_of_case_attribute_values(values: Set[Any]) -> Set[CaseAttributeValue]:
        """
        Type cast values to a set of case attribute values.

        :param values: The values to type cast.
        """
        values = make_set(values)
        if len(values) > 0 and not isinstance(next(iter(values)), CaseAttributeValue):
            values = {CaseAttributeValue(id(values), v) for v in values}
        return values

    @classmethod
    def from_obj(cls, values: Set[Any], row_obj: Optional[Any] = None) -> CaseAttribute:
        id_ = id(row_obj) if row_obj else id(values)
        values = make_set(values)
        return cls({CaseAttributeValue(id_, v) for v in values})

    @property
    def as_dict(self) -> Dict[str, Any]:
        """
        Get the case attribute as a dictionary.

        :return: The case attribute as a dictionary.
        """
        return {self.__class__.__name__: self}

    def filter_by(self, condition: CallableExpression) -> CaseAttribute:
        """
        Filter the column by a condition.

        :param condition: The condition to filter by.
        :return: The filtered column.
        """
        return self.__class__({v for v in self if condition(v)})

    def __eq__(self, other):
        if not isinstance(other, set):
            return super().__eq__(make_set(other))
        return super().__eq__(other)

    def __hash__(self):
        return hash(tuple(self.id_value_map.values()))

    def __str__(self):
        if len(self) == 0:
            return "None"
        return str({v for v in self}) if len(self) > 1 else str(next(iter(self)))

    def _to_json(self) -> Dict[str, Any]:
        return {id_: v.to_json() if isinstance(v, SubclassJSONSerializer) else v
                for id_, v in self.id_value_map.items()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> CaseAttribute:
        return cls({CaseAttributeValue.from_json(v) for id_, v in data.items()})


def create_cases_from_dataframe(df: DataFrame) -> List[Case]:
    """
    Create cases from a pandas DataFrame.

    :param df: The DataFrame to create cases from.
    :return: The cases of the DataFrame.
    """
    cases = []
    attribute_names = list(df.columns)
    for row_id, case in df.iterrows():
        case = {col_name: case[col_name].item() for col_name in attribute_names}
        cases.append(Case(_id=row_id, _type=DataFrame, **case))
    return cases


def create_case(obj: Any, recursion_idx: int = 0, max_recursion_idx: int = 0,
                obj_name: Optional[str] = None, parent_is_iterable: bool = False) -> Case:
    """
    Create a case from an object.

    :param obj: The object to create a case from.
    :param recursion_idx: The current recursion index.
    :param max_recursion_idx: The maximum recursion index to prevent infinite recursion.
    :param obj_name: The name of the object.
    :param parent_is_iterable: Boolean indicating whether the parent object is iterable or not.
    :return: The case that represents the object.
    """
    if isinstance(obj, DataFrame):
        return create_cases_from_dataframe(obj)
    if isinstance(obj, Case):
        return obj
    if ((recursion_idx > max_recursion_idx) or (obj.__class__.__module__ == "builtins")
            or (obj.__class__ in [MetaData, registry])):
        return Case(_id=id(obj), _type=obj.__class__,
                    **{obj_name or obj.__class__.__name__: make_set(obj) if parent_is_iterable else obj})
    case = Case(_id=id(obj), _type=obj.__class__)
    for attr in dir(obj):
        if attr.startswith("_") or callable(getattr(obj, attr)):
            continue
        attr_value = getattr(obj, attr)
        case = create_or_update_case_from_attribute(attr_value, attr, obj, attr, recursion_idx,
                                                    max_recursion_idx, parent_is_iterable, case)
    return case


def create_or_update_case_from_attribute(attr_value: Any, name: str, obj: Any, obj_name: Optional[str] = None,
                                         recursion_idx: int = 0, max_recursion_idx: int = 1,
                                         parent_is_iterable: bool = False,
                                         case: Optional[Case] = None) -> Case:
    """
    Create or update a case from an attribute of the object that the case represents.

    :param attr_value: The attribute value.
    :param name: The name of the attribute.
    :param obj: The parent object of the attribute.
    :param obj_name: The parent object name.
    :param recursion_idx: The recursion index to prevent infinite recursion.
    :param max_recursion_idx: The maximum recursion index.
    :param parent_is_iterable: Boolean indicating whether the parent object is iterable or not.
    :param case: The case to update.
    :return: The updated/created case.
    """
    if case is None:
        case = Case(_id=id(obj), _type=obj.__class__)
    if isinstance(attr_value, (dict, UserDict)):
        case.update({f"{obj_name}.{k}": v for k, v in attr_value.items()})
    if hasattr(attr_value, "__iter__") and not isinstance(attr_value, str):
        column = create_case_attribute_from_iterable_attribute(attr_value, name, obj, obj_name,
                                                               recursion_idx=recursion_idx + 1,
                                                               max_recursion_idx=max_recursion_idx)
        case[obj_name] = column
    else:
        case[obj_name] = make_set(attr_value) if parent_is_iterable else attr_value
    return case


def create_case_attribute_from_iterable_attribute(attr_value: Any, name: str, obj: Any, obj_name: Optional[str] = None,
                                                  recursion_idx: int = 0,
                                                  max_recursion_idx: int = 1) -> CaseAttribute:
    """
    Get a case attribute from an iterable attribute.

    :param attr_value: The iterable attribute to get the case from.
    :param name: The name of the case.
    :param obj: The parent object of the iterable.
    :param obj_name: The parent object name.
    :param recursion_idx: The recursion index to prevent infinite recursion.
    :param max_recursion_idx: The maximum recursion index.
    :return: A case attribute that represents the original iterable attribute.
    """
    values = attr_value.values() if isinstance(attr_value, (dict, UserDict)) else attr_value
    _type = type(list(values)[0]) if len(values) > 0 else get_value_type_from_type_hint(name, obj)
    attr_case = Case(_id=id(attr_value), _type=_type)
    case_attr = CaseAttribute.from_obj(values, row_obj=obj)
    for idx, val in enumerate(values):
        sub_attr_case = create_case(val, recursion_idx=recursion_idx,
                                    max_recursion_idx=max_recursion_idx,
                                    obj_name=obj_name, parent_is_iterable=True)
        attr_case.update(sub_attr_case)
    for sub_attr, val in attr_case.items():
        setattr(case_attr, sub_attr, val)
    return case_attr


def show_current_and_corner_cases(case: Any, targets: Optional[Union[List[CaseAttribute], List[SQLColumn]]] = None,
                                  current_conclusions: Optional[Union[List[CaseAttribute], List[SQLColumn]]] = None,
                                  last_evaluated_rule: Optional[Rule] = None) -> None:
    """
    Show the data of the new case and if last evaluated rule exists also show that of the corner case.

    :param case: The new case.
    :param targets: The target attribute of the case.
    :param current_conclusions: The current conclusions of the case.
    :param last_evaluated_rule: The last evaluated rule in the RDR.
    """
    corner_case = None
    if targets:
        targets = targets if isinstance(targets, list) else [targets]
    if current_conclusions:
        current_conclusions = current_conclusions if isinstance(current_conclusions, list) else [current_conclusions]
    targets = {f"target_{t.__class__.__name__}": t for t in targets} if targets else {}
    current_conclusions = {c.__class__.__name__: c for c in current_conclusions} if current_conclusions else {}
    if last_evaluated_rule:
        action = "Refinement" if last_evaluated_rule.fired else "Alternative"
        print(f"{action} needed for rule: {last_evaluated_rule}\n")
        corner_case = last_evaluated_rule.corner_case

    corner_row_dict = None
    if isinstance(case, SQLTable):
        case_dict = row_to_dict(case)
        if last_evaluated_rule and last_evaluated_rule.fired:
            corner_row_dict = row_to_dict(last_evaluated_rule.corner_case)
    else:
        case_dict = case
        if last_evaluated_rule and last_evaluated_rule.fired:
            corner_row_dict = corner_case

    if corner_row_dict:
        corner_conclusion = last_evaluated_rule.conclusion
        corner_row_dict.update({corner_conclusion.__class__.__name__: corner_conclusion})
        print(table_rows_as_str(corner_row_dict))
    print("=" * 50)
    case_dict.update(targets)
    case_dict.update(current_conclusions)
    print(table_rows_as_str(case_dict))
