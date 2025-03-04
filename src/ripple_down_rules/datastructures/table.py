from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass

from sqlalchemy.orm import DeclarativeBase as SQLTable, MappedColumn as SQLColumn
from typing_extensions import Any, Optional, Dict, Type, Set, Hashable, Tuple, Union, List, TYPE_CHECKING

from ripple_down_rules.datastructures import get_value_type_from_type_hint
from ripple_down_rules.utils import make_set, row_to_dict, table_rows_as_str

if TYPE_CHECKING:
    from ripple_down_rules.rules import Rule


class Row(UserDict):
    """
    A collection of attributes that represents a set of constraints on a case. This is a dictionary where the keys are
    the names of the attributes and the values are the attributes. All are stored in lower case.
    """

    def __init__(self, id_: Hashable, **kwargs):
        """
        Create a new row.

        :param id_: The unique identifier of the row.
        :param kwargs: The attributes of the row.
        """
        super().__init__(**kwargs)
        self.id: Hashable = id_

    def __getitem__(self, item: str) -> Any:
        return super().__getitem__(item.lower())

    def __setitem__(self, name: str, value: Any):
        name = name.lower()
        if name in self:
            if isinstance(self[name], set):
                value = make_set(value)
                if (len(self[name]) > 0 and len(value) > 0
                        and not issubclass(list(self[name])[0].__class__, list(value)[0].__class__)):
                    raise ValueError(f"Attribute {name} already exists in the case and is mutually exclusive.")
                else:
                    self[name].update(value)
            elif not issubclass(self[name].__class__, value.__class__):
                raise ValueError(f"Attribute {name} already exists in the case and is of a different type,"
                                 f"current: {self[name].__class__} != new: {value.__class__}")
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
        if not isinstance(other, (Row, dict)):
            return False
        elif isinstance(other, dict):
            return super().__eq__(Row(other))
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.id)


class ColumnMixin:
    """
    A custom set class that is used to add other attributes to the set. This is similar to a table where the set is the
    table, the attributes are the columns, and the values are the rows.
    """
    value_range: set
    """
    The range of the attribute, this can be a set of possible values or a range of numeric values (int, float).
    """
    registry: Dict[(str, type), Type[ColumnMixin]] = {}
    """
    A dictionary of all dynamically created subclasses of the CustomSet class.
    """
    nullable: bool = True
    """
    A boolean indicating whether the attribute can be None or not.
    """

    @classmethod
    def create(cls, name: str, range_: set, nullable: bool = True) -> Type[ColumnMixin]:
        """
        Create a new custom set subclass.

        :param name: The name of the column.
        :param range_: The range of the column values.
        :param nullable: Boolean indicating whether the column can be None or not.
        :return: The new column type.
        """
        existing_class = cls._get_and_update_subclass(name, range_)
        if existing_class:
            return existing_class

        new_attribute_type: Type[ColumnMixin] = type(name, (cls,), {})
        new_attribute_type.value_range = range_
        new_attribute_type.nullable = nullable

        cls.register(new_attribute_type)
        return new_attribute_type

    @classmethod
    def _get_and_update_subclass(cls, name: str, range_: set) -> Optional[Type[ColumnMixin]]:
        """
        Get a subclass of the attribute class and update its range if necessary.

        :param name: The name of the column.
        :param range_: The range of the column values.
        """
        key = (name, cls)
        if key in cls.registry:
            if not cls.registry[key].is_within_range(range_):
                if isinstance(cls.registry[key].value_range, set):
                    cls.registry[key].value_range.update(range_)
                else:
                    raise ValueError(f"Range of {key} is different from {cls.registry[key].value_range}.")
            return cls.registry[key]

    @classmethod
    def register(cls, subclass: Type[ColumnMixin]):
        """
        Register a subclass of the attribute class, this is used to be able to dynamically create Attribute subclasses.

        :param subclass: The subclass to register.
        """
        if not issubclass(subclass, ColumnMixin):
            raise ValueError(f"{subclass} is not a subclass of CustomSet.")
        if subclass not in cls.registry:
            cls.registry[subclass.__name__.lower()] = subclass
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
            return set(value).issubset(cls.value_range)
        elif isinstance(value, str):
            return value.lower() in cls.value_range
        else:
            return value in cls.value_range


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
        return self.id == other.id and self.value == other.value

    def __hash__(self):
        return self.id


class Column(set, ColumnMixin):
    def __init__(self, values: Set[ColumnValue]):
        """
        Create a new column.

        :param values: The values of the column.
        """
        values = make_set(values)
        self.id_value_map: Dict[Hashable, Set[ColumnValue]] = {id(v): v for v in values}
        super().__init__([v.value for v in values])

    def __getitem__(self, row_id: Hashable) -> Set[ColumnValue]:
        return {v for v in self if v.id == row_id}

    @classmethod
    def from_obj(cls, row_obj: Any, values: Set[Any]):
        return cls({ColumnValue(id(row_obj), v) for v in values})

    def add(self, value: ColumnValue):
        self.id_value_map[id(value)] = value
        super().add(value.value)

    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return hash(tuple(self.id_value_map.values()))

    def __str__(self):
        return str({v for v in self})


class Table(Row, ColumnMixin):
    """
    A table is a collection of rows that are all of the same type. This is similar to a database table where each row
    is a case and each column is an attribute of the case.
    """
    id_row_map: Dict[Hashable, Row]
    """
    A dictionary of all rows in the table where the key is the id of the row and the value is the row.
    """

    def __init__(self, id_: Hashable, rows: Optional[Set[Row]] = None):
        """
        Create a new table.

        :param id_: The unique identifier of the table.
        :param rows: The rows of the table.
        """
        Row.__init__(self, id_)
        self.id_row_map = {}
        self.update(rows if rows else {})

    def update(self, *rows):
        for row in rows:
            if len(row) == 0:
                continue
            if isinstance(row, Row):
                self.assert_is_new_row(row)
                self.assert_row_has_required_columns(row)
                self.id_row_map[row.id] = row
            else:
                row = create_table(row)
            super().update(row)

    def assert_is_new_row(self, row: Row):
        """
        Check if a row is new.

        :param row: The row to check.
        """
        if row.id in self.id_row_map:
            raise ValueError(f"Row with id {row.id} already exists in the table.")

    def assert_row_has_required_columns(self, row: Row):
        """
        Check if a row has all the required columns.
        """
        for column_name in self.column_names:
            if column_name not in row and not self[column_name].nullable:
                raise ValueError(f"Row {row} is missing attribute {column_name}.")

    @property
    def column_names(self) -> Set[str]:
        """
        Get all column names of the table.

        :return: A set of all column names.
        """
        return {column_name for column_name in self.keys()}

    @property
    def columns(self) -> Dict[str, Column]:
        """
        Get all columns of the table.

        :return: A set of all columns.
        """
        return {column_name: self[column_name] for column_name in self.column_names}

    def create_and_add_column(self, name: str, range_: set, values: Set[ColumnValue]):
        """
        Add a new column to the table.

        :param name: The name of the column.
        :param range_: The range of the column.
        :param values: The values of the column.
        """
        column_type = Column.create(name, range_)
        self.add_column(column_type(values))

    def add_column(self, column: Column):
        """
        Add a new column to the table.

        :param column: The column to add.
        """
        if 0 < len(self) != len(column):
            raise ValueError(f"Column length {len(column)} does not match the number of rows {len(self)}.")
        if len(self) == 0:
            self[column.__class__.__name__] = column
        else:
            for row in self.id_row_map.values():
                row[column.__class__.__name__] = column[row.id]

    def __eq__(self, other):
        if not isinstance(other, Table):
            return False
        return Row.__eq__(self, other)

    def __hash__(self):
        return self.id

    def __str__(self):
        return Row.__str__(self)


def create_table(obj: Any, recursion_idx: int = 0, max_recursion_idx: int = 1,
                 obj_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a table from an object.

    :param obj: The object to create a table from.
    :param recursion_idx: The current recursion index.
    :param max_recursion_idx: The maximum recursion index to prevent infinite recursion.
    :param obj_name: The name of the object.
    :return: The table of the object.
    """
    if hasattr(obj, "__iter__") and not isinstance(obj, str):
        obj_name = obj_name or obj.__class__.__name__
        values = list(obj.values()) if isinstance(obj, (dict, UserDict)) else obj
        table = Table.create(obj_name, make_set(type(values[0])))(id_=id(obj))
        if isinstance(obj, (dict, UserDict)):
            for k, v in obj.items():
                table.create_and_add_column(k, make_set(type(v)), {ColumnValue(id(obj), v)})
        else:
            for v in values:
                table.update(create_table(v, recursion_idx=recursion_idx + 1,
                                          max_recursion_idx=max_recursion_idx, obj_name=obj_name))
        return table
    table = Table.create(obj.__class__.__name__, make_set(obj.__class__))(id_=id(obj))
    if recursion_idx > max_recursion_idx:
        return table
    for attr in dir(obj):
        if attr.startswith("_") or callable(getattr(obj, attr)):
            continue
        attr_value = getattr(obj, attr)
        chained_name = f"{obj_name}.{attr}" if obj_name else attr
        if isinstance(attr_value, (dict, UserDict)):
            table.update({f"{chained_name}.{k}": v for k, v in attr_value.items()})
        column, column_ref_table = get_ref_column_and_its_table(attr_value, attr, obj, chained_name,
                                                                recursion_idx=recursion_idx + 1,
                                                                max_recursion_idx=max_recursion_idx)
        update_table_with_column_and_its_table(table, column, column_ref_table, chained_name)
    return table


def update_table_with_column_and_its_table(table: Table, column: Column, column_ref_table: Table, column_name: str):
    """
    Update a table with a column and its table.

    :param table: The table to update.
    :param column: The column to add.
    :param column_ref_table: The table of the column.
    :param column_name: The name of the column.
    """
    for sub_column_name, val in column_ref_table.items():
        setattr(column, sub_column_name.replace(f"{column_name}.", ""), val)
    table[column_name] = column
    table.update(column_ref_table)


def get_ref_column_and_its_table(attr_value: Any, name: str, obj: Any, obj_name: Optional[str] = None,
                                 recursion_idx: int = 0, max_recursion_idx: int = 1) -> Tuple[Column, Table]:
    """
    Get a reference column and its table.

    :param attr_value: The attribute value to get the column and table from.
    :param name: The name of the attribute.
    :param obj: The parent object of the attribute.
    :param obj_name: The parent object name.
    :param recursion_idx: The recursion index to prevent infinite recursion.
    :param max_recursion_idx: The maximum recursion index.
    :return: A reference column and its table.
    """
    if hasattr(attr_value, "__iter__") and not isinstance(attr_value, str):
        column, values_table = get_table_from_iterable_attribute(attr_value, name, obj, obj_name,
                                                                 recursion_idx=recursion_idx,
                                                                 max_recursion_idx=max_recursion_idx)
    else:
        column = Column.create(name, {type(attr_value)}).from_obj(obj, make_set(attr_value))
        values_table = create_table(attr_value, recursion_idx=recursion_idx,
                                    max_recursion_idx=max_recursion_idx, obj_name=obj_name)
    return column, values_table


def get_table_from_iterable_attribute(attr_value: Any, name: str, obj: Any, obj_name: Optional[str] = None,
                                      recursion_idx: int = 0, max_recursion_idx: int = 1) -> Tuple[Column, Table]:
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
    column = Column.create(name, range_).from_obj(obj, values)
    table_type = Table.create(name, range_)
    values_table = table_type(id_=id(values))
    for idx, val in enumerate(values):
        val_table = create_table(val, recursion_idx=recursion_idx,
                                 max_recursion_idx=max_recursion_idx,
                                 obj_name=obj_name)
        values_table.update(val_table)
    return column, values_table


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
        case_dict = create_table(case, max_recursion_idx=0)
        if last_evaluated_rule and last_evaluated_rule.fired:
            corner_row_dict = create_table(corner_case, max_recursion_idx=0)

    if corner_row_dict:
        corner_conclusion = last_evaluated_rule.conclusion
        corner_row_dict.update({corner_conclusion.__class__.__name__: corner_conclusion})
        print(table_rows_as_str(corner_row_dict))
    print("=" * 50)
    case_dict.update(targets)
    case_dict.update(current_conclusions)
    print(table_rows_as_str(case_dict))
