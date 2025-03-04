from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass

from pandas import DataFrame
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase as SQLTable, MappedColumn as SQLColumn, registry
from typing_extensions import Any, Optional, Dict, Type, Set, Hashable, Union, List, TYPE_CHECKING, Tuple

from . import get_value_type_from_type_hint
from ..utils import make_set, row_to_dict, table_rows_as_str

if TYPE_CHECKING:
    from ripple_down_rules.rules import Rule


class SubClassFactory:
    """
    A custom set class that is used to add other attributes to the set. This is similar to a table where the set is the
    table, the attributes are the columns, and the values are the rows.
    """
    _value_range: set
    """
    The range of the attribute, this can be a set of possible values or a range of numeric values (int, float).
    """
    _registry: Dict[(str, type), Type[SubClassFactory]] = {}
    """
    A dictionary of all dynamically created subclasses of this class.
    """

    @classmethod
    def create(cls, name: str, range_: set, **kwargs) -> Type[SubClassFactory]:
        """
        Create a new custom set subclass.

        :param name: The name of the column.
        :param range_: The range of the column values.
        :return: The new column type.
        """
        existing_class = cls._get_and_update_subclass(name, range_)
        if existing_class:
            return existing_class

        new_attribute_type: Type[SubClassFactory] = type(name, (cls,), {})
        new_attribute_type._value_range = range_
        for key, value in kwargs.items():
            setattr(new_attribute_type, key, value)

        cls.register(new_attribute_type)
        return new_attribute_type

    @classmethod
    def _get_and_update_subclass(cls, name: str, range_: set) -> Optional[Type[SubClassFactory]]:
        """
        Get a subclass of the attribute class and update its range if necessary.

        :param name: The name of the column.
        :param range_: The range of the column values.
        """
        key = (name.lower(), cls)
        if key in cls._registry:
            if not cls._registry[key].is_within_range(range_):
                if isinstance(cls._registry[key]._value_range, set):
                    cls._registry[key]._value_range.update(range_)
                else:
                    raise ValueError(f"Range of {key} is different from {cls._registry[key]._value_range}.")
            return cls._registry[key]

    @classmethod
    def register(cls, subclass: Type[SubClassFactory]):
        """
        Register a subclass of the attribute class, this is used to be able to dynamically create Attribute subclasses.

        :param subclass: The subclass to register.
        """
        if not issubclass(subclass, SubClassFactory):
            raise ValueError(f"{subclass} is not a subclass of CustomSet.")
        if subclass not in cls._registry:
            cls._registry[(subclass.__name__.lower(), cls)] = subclass
        else:
            raise ValueError(f"{subclass} is already registered.")

    @classmethod
    def is_within_range(cls, value: Any) -> bool:
        """
        Check if a value is within the range of the custom set.

        :param value: The value to check.
        :return: Boolean indicating whether the value is within the range or not.
        """
        if hasattr(value, "__iter__") and not isinstance(value, str):
            return set(value).issubset(cls._value_range)
        elif isinstance(value, str):
            return value.lower() in cls._value_range
        else:
            return value in cls._value_range

    def __instancecheck__(self, instance):
        return isinstance(instance, (SubClassFactory, *self._value_range))


class Row(UserDict, SubClassFactory):
    """
    A collection of attributes that represents a set of constraints on a case. This is a dictionary where the keys are
    the names of the attributes and the values are the attributes. All are stored in lower case.
    """

    def __getitem__(self, item: str) -> Any:
        return super().__getitem__(item.lower())

    def __setitem__(self, name: str, value: Any):
        name = name.lower()
        if name in self:
            if isinstance(self[name], set):
                self[name].update(make_set(value))
            else:
                raise ValueError(f"Attribute {name} already exists in the case and is mutually exclusive.")
        else:
            setattr(self, name, value)
            super().__setitem__(name, value)

    def __contains__(self, item):
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
        return hash(tuple(self.items()))

    def __instancecheck__(self, instance):
        return isinstance(instance, (dict, UserDict, Row)) or super().__instancecheck__(instance)


@dataclass
class ColumnValue:
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


class Column(set, SubClassFactory):
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
        values = make_set(values)
        self.id_value_map: Dict[Hashable, Union[ColumnValue, Set[ColumnValue]]] = {id(v): v for v in values}
        super().__init__([v.value for v in values])

    @classmethod
    def create(cls, name: str, range_: set,
               nullable: bool = False, mutually_exclusive: bool = False) -> Type[SubClassFactory]:
        return super().create(name, range_, **{"nullable": nullable, "mutually_exclusive": mutually_exclusive})

    @classmethod
    def from_obj(cls, row_obj: Any, values: Set[Any]):
        return cls({ColumnValue(id(row_obj), v) for v in values})

    def __eq__(self, other):
        if not isinstance(other, set):
            return super().__eq__(make_set(other))
        return super().__eq__(other)

    def __hash__(self):
        return hash(tuple(self.id_value_map.values()))

    def __str__(self):
        return str({v for v in self})

    def __instancecheck__(self, instance):
        return isinstance(instance, (set, self.__class__)) or super().__instancecheck__(instance)


def create_row_from_dataframe(df: DataFrame, name: Optional[str] = None) -> Row:
    """
    Create a row from a pandas DataFrame.

    :param df: The DataFrame to create a row from.
    :param name: The name of the DataFrame.
    :return: The row of the DataFrame.
    """
    row = Row()
    for col in df.columns:
        range_ = {type(list(df[col].iterrows())[0])} if len(df[col].iterrows()) > 0 else set()
        row[col] = Column.create(col, range_)(list(df[col].iterrows()))
    return row


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
    row = Row.create(obj_name or obj.__class__.__name__, make_set(type(obj)))()
    if ((recursion_idx > max_recursion_idx) or (obj.__class__.__module__ == "builtins")
            or (obj.__class__ in [MetaData, registry])):
        return row
    for attr in dir(obj):
        if attr.startswith("_") or callable(getattr(obj, attr)):
            continue
        attr_value = getattr(obj, attr)
        chained_name = f"{obj_name}.{attr}" if obj_name else attr
        row = create_or_update_row_from_attribute(attr_value, attr, obj, chained_name, recursion_idx,
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
    row = row if row is not None else Row.create(obj_name or obj.__class__.__name__, make_set(type(obj)))()
    if isinstance(attr_value, (dict, UserDict)):
        row.update({f"{obj_name}.{k}": v for k, v in attr_value.items()})
    if hasattr(attr_value, "__iter__") and not isinstance(attr_value, str):
        column, attr_row = create_column_and_row_from_iterable_attribute(attr_value, name, obj, obj_name,
                                                                         recursion_idx=recursion_idx + 1,
                                                                         max_recursion_idx=max_recursion_idx)
        row[obj_name] = column
    else:
        attr_row = create_row(attr_value, recursion_idx=recursion_idx + 1,
                              max_recursion_idx=max_recursion_idx,
                              obj_name=obj_name, parent_is_iterable=False)
        row[obj_name] = make_set(attr_value) if parent_is_iterable else attr_value
    row.update(attr_row)
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
    attr_row = Row.create(name or list(range_)[0].__name__, range_)()
    column = Column.create(name, range_).from_obj(obj, values)
    for idx, val in enumerate(values):
        sub_attr_row = create_row(val, recursion_idx=recursion_idx,
                                  max_recursion_idx=max_recursion_idx,
                                  obj_name=obj_name, parent_is_iterable=True)
        attr_row.update(sub_attr_row)
    for sub_attr, val in attr_row.items():
        setattr(column, sub_attr.replace(f"{obj_name}.", ""), val)
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
        case_dict = create_row(case, max_recursion_idx=0)
        if last_evaluated_rule and last_evaluated_rule.fired:
            corner_row_dict = create_row(corner_case, max_recursion_idx=0)

    if corner_row_dict:
        corner_conclusion = last_evaluated_rule.conclusion
        corner_row_dict.update({corner_conclusion.__class__.__name__: corner_conclusion})
        print(table_rows_as_str(corner_row_dict))
    print("=" * 50)
    case_dict.update(targets)
    case_dict.update(current_conclusions)
    print(table_rows_as_str(case_dict))
