from __future__ import annotations

import os
import time
from abc import ABC
from collections import UserDict
from copy import deepcopy, copy
from dataclasses import dataclass
from enum import Enum

from pandas import DataFrame
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase as SQLTable, MappedColumn as SQLColumn, registry
from typing_extensions import Any, Optional, Dict, Type, Set, Hashable, Union, List, TYPE_CHECKING, Tuple, Self

from ..utils import make_set, row_to_dict, table_rows_as_str, get_value_type_from_type_hint, make_list, \
    SubclassJSONSerializer

if TYPE_CHECKING:
    from ripple_down_rules.rules import Rule
    from .callable_expression import CallableExpression


class Row(UserDict, SubclassJSONSerializer):
    """
    A collection of attributes that represents a set of constraints on a case. This is a dictionary where the keys are
    the names of the attributes and the values are the attributes. All are stored in lower case.
    """

    def __init__(self, id_: Optional[Hashable] = None, **kwargs):
        """
        Create a new row.

        :param id_: The id of the row.
        :param kwargs: The attributes of the row.
        """
        super().__init__(kwargs)
        self.id_ = id_ if id_ else id(self)

    @classmethod
    def from_obj(cls, obj: Any, obj_name: Optional[str] = None, max_recursion_idx: int = 3) -> Row:
        """
        Create a row from an object.

        :param obj: The object to create a row from.
        :param max_recursion_idx: The maximum recursion index to prevent infinite recursion.
        :param obj_name: The name of the object.
        :return: The row of the object.
        """
        return create_row(obj, max_recursion_idx=max_recursion_idx, obj_name=obj_name)

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

    def __eq__(self, other):
        if not isinstance(other, (Row, dict, UserDict)):
            return False
        elif isinstance(other, (dict, UserDict)):
            return super().__eq__(Row(other))
        else:
            return super().__eq__(other)

    def __hash__(self):
        return self.id_

    def _to_json(self) -> Dict[str, Any]:
        serializable = {k: v for k, v in self.items() if not k.startswith("_")}
        serializable["_id"] = self.id_
        return {k: v.to_json() if isinstance(v, SubclassJSONSerializer) else v for k, v in serializable.items()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Row:
        id_ = data.pop("_id")
        return cls(id_=id_, **data)


@dataclass
class ColumnValue(SubclassJSONSerializer):
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
        if not isinstance(other, ColumnValue):
            return False
        return self.value == other.value

    def __hash__(self):
        return self.id

    def _to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "value": self.value}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> ColumnValue:
        return cls(id=data["id"], value=data["value"])


class Column(set, SubclassJSONSerializer):
    nullable: bool = True
    """
    A boolean indicating whether the column can be None or not.
    """
    mutually_exclusive: bool = False
    """
    A boolean indicating whether the column is mutually exclusive or not. (i.e. can only have one value)
    """

    def __init__(self, values: Set[ColumnValue]):
        """
        Create a new column.

        :param values: The values of the column.
        """
        values = self._type_cast_values_to_set_of_column_values(values)
        self.id_value_map: Dict[Hashable, Union[ColumnValue, Set[ColumnValue]]] = {id(v): v for v in values}
        super().__init__([v.value for v in values])

    @staticmethod
    def _type_cast_values_to_set_of_column_values(values: Set[Any]) -> Set[ColumnValue]:
        """
        Type cast values to a set of column values.

        :param values: The values to type cast.
        """
        values = make_set(values)
        if len(values) > 0 and not isinstance(next(iter(values)), ColumnValue):
            values = {ColumnValue(id(values), v) for v in values}
        return values

    @classmethod
    def from_obj(cls, values: Set[Any], row_obj: Optional[Any] = None) -> Column:
        id_ = id(row_obj) if row_obj else id(values)
        values = make_set(values)
        return cls({ColumnValue(id_, v) for v in values})

    @property
    def as_dict(self) -> Dict[str, Any]:
        """
        Get the column as a dictionary.

        :return: The column as a dictionary.
        """
        return {self.__class__.__name__: self}

    def filter_by(self, condition: CallableExpression) -> Column:
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
    def _from_json(cls, data: Dict[str, Any]) -> Column:
        return cls({ColumnValue.from_json(v) for id_, v in data.items()})


def create_rows_from_dataframe(df: DataFrame, name: Optional[str] = None) -> List[Row]:
    """
    Create a row from a pandas DataFrame.

    :param df: The DataFrame to create a row from.
    :param name: The name of the DataFrame.
    :return: The row of the DataFrame.
    """
    rows = []
    col_names = list(df.columns)
    for row_id, row in df.iterrows():
        row = {col_name: row[col_name].item() for col_name in col_names}
        rows.append(Row(id_=row_id, **row))
    return rows


def create_row(obj: Any, recursion_idx: int = 0, max_recursion_idx: int = 0,
               obj_name: Optional[str] = None, parent_is_iterable: bool = False) -> Row:
    """
    Create a table from an object.

    :param obj: The object to create a table from.
    :param recursion_idx: The current recursion index.
    :param max_recursion_idx: The maximum recursion index to prevent infinite recursion.
    :param obj_name: The name of the object.
    :param parent_is_iterable: Boolean indicating whether the parent object is iterable or not.
    :return: The table of the object.
    """
    if isinstance(obj, Row):
        return obj
    if ((recursion_idx > max_recursion_idx) or (obj.__class__.__module__ == "builtins")
            or (obj.__class__ in [MetaData, registry])):
        return Row(id_=id(obj), **{obj_name or obj.__class__.__name__: make_set(obj) if parent_is_iterable else obj})
    row = Row(id_=id(obj))
    for attr in dir(obj):
        if attr.startswith("_") or callable(getattr(obj, attr)):
            continue
        attr_value = getattr(obj, attr)
        row = create_or_update_row_from_attribute(attr_value, attr, obj, attr, recursion_idx,
                                                  max_recursion_idx, parent_is_iterable, row)
    return row


def create_or_update_row_from_attribute(attr_value: Any, name: str, obj: Any, obj_name: Optional[str] = None,
                                        recursion_idx: int = 0, max_recursion_idx: int = 1,
                                        parent_is_iterable: bool = False,
                                        row: Optional[Row] = None) -> Row:
    """
    Get a reference column and its table.

    :param attr_value: The attribute value to get the column and table from.
    :param name: The name of the attribute.
    :param obj: The parent object of the attribute.
    :param obj_name: The parent object name.
    :param recursion_idx: The recursion index to prevent infinite recursion.
    :param max_recursion_idx: The maximum recursion index.
    :param parent_is_iterable: Boolean indicating whether the parent object is iterable or not.
    :param row: The row to update.
    :return: A reference column and its table.
    """
    if row is None:
        row = Row(id_=id(obj))
    if isinstance(attr_value, (dict, UserDict)):
        row.update({f"{obj_name}.{k}": v for k, v in attr_value.items()})
    if hasattr(attr_value, "__iter__") and not isinstance(attr_value, str):
        column, attr_row = create_column_and_row_from_iterable_attribute(attr_value, name, obj, obj_name,
                                                                         recursion_idx=recursion_idx + 1,
                                                                         max_recursion_idx=max_recursion_idx)
        row[obj_name] = column
    else:
        row[obj_name] = make_set(attr_value) if parent_is_iterable else attr_value
    return row


def create_column_and_row_from_iterable_attribute(attr_value: Any, name: str, obj: Any, obj_name: Optional[str] = None,
                                                  recursion_idx: int = 0,
                                                  max_recursion_idx: int = 1) -> Tuple[Column, Row]:
    """
    Get a table from an iterable.

    :param attr_value: The iterable attribute to get the table from.
    :param name: The name of the table.
    :param obj: The parent object of the iterable.
    :param obj_name: The parent object name.
    :param recursion_idx: The recursion index to prevent infinite recursion.
    :param max_recursion_idx: The maximum recursion index.
    :return: A table of the iterable.
    """
    values = attr_value.values() if isinstance(attr_value, (dict, UserDict)) else attr_value
    range_ = {type(list(values)[0])} if len(values) > 0 else set()
    if len(range_) == 0:
        range_ = make_set(get_value_type_from_type_hint(name, obj))
    if not range_:
        raise ValueError(f"Could not determine the range of {name} in {obj}.")
    attr_row = Row(id_=id(attr_value))
    column = Column.from_obj(values, row_obj=obj)
    for idx, val in enumerate(values):
        sub_attr_row = create_row(val, recursion_idx=recursion_idx,
                                  max_recursion_idx=max_recursion_idx,
                                  obj_name=obj_name, parent_is_iterable=True)
        attr_row.update(sub_attr_row)
    for sub_attr, val in attr_row.items():
        setattr(column, sub_attr, val)
    return column, attr_row


def show_current_and_corner_cases(case: Any, targets: Optional[Union[List[Column], List[SQLColumn]]] = None,
                                  current_conclusions: Optional[Union[List[Column], List[SQLColumn]]] = None,
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


Case = Row
