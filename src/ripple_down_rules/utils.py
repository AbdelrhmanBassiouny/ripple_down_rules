from __future__ import annotations

import ast
import builtins
import importlib
import json
import logging
import os
import re
from collections import UserDict
from copy import deepcopy
from dataclasses import is_dataclass, fields
from enum import Enum
from types import NoneType

import matplotlib
import networkx as nx
import requests
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from matplotlib import pyplot as plt
from sqlalchemy import MetaData, inspect
from sqlalchemy.orm import Mapped, registry, class_mapper, DeclarativeBase as SQLTable, Session
from tabulate import tabulate
from typing_extensions import Callable, Set, Any, Type, Dict, TYPE_CHECKING, get_type_hints, \
    get_origin, get_args, Tuple, Optional, List, Union, Self

if TYPE_CHECKING:
    from .datastructures.case import Case
    from .datastructures.dataclasses import CaseQuery
    from .rules import Rule

import ast

matplotlib.use("Qt5Agg")  # or "Qt5Agg", depending on availability


def encapsulate_user_input(user_input: str, func_signature: str) -> str:
    """
    Encapsulate the user input string with a function definition.

    :param user_input: The user input string.
    :param func_signature: The function signature to use for encapsulation.
    :return: The encapsulated user input string.
    """
    if func_signature not in user_input:
        new_user_input = func_signature + "\n    "
        if "return " not in user_input:
            if '\n' not in user_input:
                new_user_input += f"return {user_input}"
            else:
                raise ValueError("User input must contain a return statement or be a single line.")
        else:
            for cl in user_input.split('\n'):
                sub_code_lines = cl.split('\n')
                new_user_input += '\n    '.join(sub_code_lines) + '\n    '
    else:
        new_user_input = user_input
    return new_user_input


def build_user_input_from_conclusion(conclusion: Any) -> str:
    """
    Build a user input string from the conclusion.

    :param conclusion: The conclusion to use for the callable expression.
    :return: The user input string.
    """

    # set user_input to the string representation of the conclusion
    if isinstance(conclusion, set):
        user_input = '{' + f"{', '.join([conclusion_to_str(t) for t in conclusion])}" + '}'
    elif isinstance(conclusion, list):
        user_input = '[' + f"{', '.join([conclusion_to_str(t) for t in conclusion])}" + ']'
    elif isinstance(conclusion, tuple):
        user_input = '(' + f"{', '.join([conclusion_to_str(t) for t in conclusion])}" + ')'
    else:
        user_input = conclusion_to_str(conclusion)

    return user_input


def conclusion_to_str(conclusion_: Any) -> str:
    if isinstance(conclusion_, Enum):
        return type(conclusion_).__name__ + '.' + conclusion_.name
    else:
        return str(conclusion_)


def update_case(case_query: CaseQuery, conclusions: Dict[str, Any]):
    """
    Update the case with the conclusions.

    :param case_query: The case query that contains the case to update.
    :param conclusions: The conclusions to update the case with.
    """
    if not conclusions:
        return
    if len(conclusions) == 0:
        return
    if isinstance(case_query.original_case, SQLTable) or is_dataclass(case_query.original_case):
        for conclusion_name, conclusion in conclusions.items():
            attribute = getattr(case_query.case, conclusion_name)
            if conclusion_name == case_query.attribute_name:
                attribute_type = case_query.attribute_type
            else:
                attribute_type = (get_case_attribute_type(case_query.original_case, conclusion_name, attribute),)
            if isinstance(attribute, set):
                for c in conclusion:
                    attribute.update(make_set(c))
            elif isinstance(attribute, list):
                attribute.extend(conclusion)
            elif any(at in {List, list} for at in attribute_type):
                attribute = [] if attribute is None else attribute
                attribute.extend(conclusion)
            elif any(at in {Set, set} for at in attribute_type):
                attribute = set() if attribute is None else attribute
                for c in conclusion:
                    attribute.update(make_set(c))
            elif is_iterable(conclusion) and len(conclusion) == 1 \
                    and any(at is type(list(conclusion)[0]) for at in attribute_type):
                setattr(case_query.case, conclusion_name, list(conclusion)[0])
            elif not is_iterable(conclusion) and any(at is type(conclusion) for at in attribute_type):
                setattr(case_query.case, conclusion_name, conclusion)
            else:
                raise ValueError(f"Unknown type or type mismatch for attribute {conclusion_name} with type "
                                 f"{case_query.attribute_type} with conclusion "
                                 f"{conclusion} of type {type(conclusion)}")
    else:
        case_query.case.update(conclusions)


def is_conflicting(conclusion: Any, target: Any) -> bool:
    """
    :param conclusion: The conclusion to check.
    :param target: The target to compare the conclusion with.
    :return: Whether the conclusion is conflicting with the target by have different values for same type categories.
    """
    return have_common_types(conclusion, target) and not make_set(conclusion).issubset(make_set(target))


def have_common_types(conclusion: Any, target: Any) -> bool:
    """
    :param conclusion: The conclusion to check.
    :param target: The target to compare the conclusion with.
    :return: Whether the conclusion shares some types with the target.
    """
    target_types = {type(t) for t in make_set(target)}
    conclusion_types = {type(c) for c in make_set(conclusion)}
    common_types = conclusion_types.intersection(target_types)
    return len(common_types) > 0


def calculate_precision_and_recall(pred_cat: Dict[str, Any], target: Dict[str, Any]) -> Tuple[
    List[bool], List[bool]]:
    """
    :param pred_cat: The predicted category.
    :param target: The target category.
    :return: The precision and recall of the classifier.
    """
    recall = []
    precision = []
    for pred_key, pred_value in pred_cat.items():
        if pred_key not in target:
            continue
        precision.extend([v in make_set(target[pred_key]) for v in make_set(pred_value)])
    for target_key, target_value in target.items():
        if target_key not in pred_cat:
            recall.append(False)
            continue
        recall.extend([v in make_set(pred_cat[target_key]) for v in make_set(target_value)])
    return precision, recall


def get_rule_conclusion_as_source_code(rule: Rule, conclusion: str, parent_indent: str = "") -> Tuple[str, str]:
    """
    Convert the conclusion of a rule to source code.

    :param rule: The rule to get the conclusion from.
    :param conclusion: The conclusion to convert to source code.
    :param parent_indent: The indentation to use for the source code.
    :return: The source code of the conclusion as a tuple of strings, one for the function and one for the call.
    """
    indent = f"{parent_indent}{' ' * 4}"
    if "def " in conclusion:
        # This means the conclusion is a definition that should be written and then called
        conclusion_lines = conclusion.split('\n')
        # use regex to replace the function name
        new_function_name = f"def conclusion_{id(rule)}"
        conclusion_lines[0] = re.sub(r"def (\w+)", new_function_name, conclusion_lines[0])
        func_call = f"{indent}return {new_function_name.replace('def ', '')}(case)\n"
        return "\n".join(conclusion_lines).strip(' '), func_call
    else:
        raise ValueError(f"Conclusion is format is not valid, it should be a one line string or "
                         f"contain a function definition. Instead got:\n{conclusion}\n")


def ask_llm(prompt):
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "codellama:7b-instruct",  # or "phi"
            "prompt": prompt,
            "stream": False,
        })
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"❌ Local LLM error: {e}"


def get_case_attribute_type(original_case: Any, attribute_name: str,
                            known_value: Optional[Any] = None) -> Type:
    """
    :param original_case: The case to get the attribute from.
    :param attribute_name: The name of the attribute.
    :param known_value: A known value of the attribute.
    :return: The type of the attribute.
    """
    if known_value is not None:
        return type(known_value)
    elif hasattr(original_case, attribute_name):
        hint, origin, args = get_hint_for_attribute(attribute_name, original_case)
        if origin is not None:
            origin = typing_to_python_type(origin)
        if origin == Union:
            if len(args) == 2:
                if args[1] is type(None):
                    return typing_to_python_type(args[0])
                elif args[0] is type(None):
                    return typing_to_python_type(args[1])
            elif len(args) == 1:
                return typing_to_python_type(args[0])
            else:
                raise ValueError(f"Union with more than 2 types is not supported: {args}")
        elif origin is not None:
            return origin
        if hint is not None:
            return typing_to_python_type(hint)


def conclusion_to_json(conclusion):
    if is_iterable(conclusion):
        conclusions = {'_type': get_full_class_name(type(conclusion)), 'value': []}
        for c in conclusion:
            conclusions['value'].append(conclusion_to_json(c))
    elif hasattr(conclusion, 'to_json'):
        conclusions = conclusion.to_json()
    else:
        conclusions = {'_type': get_full_class_name(type(conclusion)), 'value': conclusion}
    return conclusions


def contains_return_statement(source: str) -> bool:
    """
    :param source: The source code to check.
    :return: True if the source code contains a return statement, False otherwise.
    """
    try:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Return):
                return True
        return False
    except SyntaxError:
        return False


def get_names_used(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def extract_dependencies(code_lines):
    full_code = '\n'.join(code_lines)
    tree = ast.parse(full_code)
    final_stmt = tree.body[-1]

    if not isinstance(final_stmt, ast.Return):
        raise ValueError("Last line is not a return statement")
    if final_stmt.value is None:
        raise ValueError("Return statement has no value")
    needed = get_names_used(final_stmt.value)
    required_lines = []
    line_map = {id(node): i for i, node in enumerate(tree.body)}

    def handle_stmt(stmt, needed):
        keep = False
        if isinstance(stmt, ast.Assign):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            if any(t in needed for t in targets):
                needed.update(get_names_used(stmt.value))
                keep = True
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id in needed:
                needed.update(get_names_used(stmt.value))
                keep = True
        elif isinstance(stmt, ast.FunctionDef):
            if stmt.name in needed:
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Name):
                        needed.add(n.id)
                keep = True
        elif isinstance(stmt, (ast.For, ast.While, ast.If)):
            # Check if any of the body statements interact with needed variables
            for substmt in stmt.body + getattr(stmt, 'orelse', []):
                if handle_stmt(substmt, needed):
                    keep = True
            # Also check the condition (test or iter)
            if isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name) and stmt.target.id in needed:
                    keep = True
                needed.update(get_names_used(stmt.iter))
            elif isinstance(stmt, ast.If) or isinstance(stmt, ast.While):
                needed.update(get_names_used(stmt.test))

        return keep

    for stmt in reversed(tree.body[:-1]):
        if handle_stmt(stmt, needed):
            required_lines.insert(0, code_lines[line_map[id(stmt)]])

    required_lines.append(code_lines[-1])  # Always include return
    return required_lines


def serialize_dataclass(obj: Any) -> Union[Dict, Any]:
    """
    Recursively serialize a dataclass to a dictionary. If the dataclass contains any nested dataclasses, they will be
    serialized as well. If the object is not a dataclass, it will be returned as is.

    :param obj: The dataclass to serialize.
    :return: The serialized dataclass as a dictionary or the object itself if it is not a dataclass.
    """

    def recursive_convert(obj):
        if is_dataclass(obj):
            return {
                "__dataclass__": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "fields": {f.name: recursive_convert(getattr(obj, f.name)) for f in fields(obj)}
            }
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        else:
            return obj

    return recursive_convert(obj)


def deserialize_dataclass(data: dict) -> Any:
    """
    Recursively deserialize a dataclass from a dictionary, if the dictionary contains a key "__dataclass__" (Most likely
    created by the serialize_dataclass function), it will be treated as a dataclass and deserialized accordingly,
    otherwise it will be returned as is.

    :param data: The dictionary to deserialize.
    :return: The deserialized dataclass.
    """

    def recursive_load(obj):
        if isinstance(obj, dict) and "__dataclass__" in obj:
            module_name, class_name = obj["__dataclass__"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls: Type = getattr(module, class_name)
            field_values = {
                k: recursive_load(v)
                for k, v in obj["fields"].items()
            }
            return cls(**field_values)
        elif isinstance(obj, list):
            return [recursive_load(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: recursive_load(v) for k, v in obj.items()}
        else:
            return obj

    return recursive_load(data)


def typing_to_python_type(typing_hint: Type) -> Type:
    """
    Convert a typing hint to a python type.

    :param typing_hint: The typing hint to convert.
    :return: The python type.
    """
    if typing_hint in [list, List]:
        return list
    elif typing_hint in [tuple, Tuple]:
        return tuple
    elif typing_hint in [set, Set]:
        return set
    elif typing_hint in [dict, Dict]:
        return dict
    else:
        return typing_hint


def capture_variable_assignment(code: str, variable_name: str) -> Optional[str]:
    """
    Capture the assignment of a variable in the given code.

    :param code: The code to analyze.
    :param variable_name: The name of the variable to capture.
    :return: The assignment statement or None if not found.
    """
    tree = ast.parse(code)
    assignment = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    # now extract the right side of the assignment
                    assignment = ast.get_source_segment(code, node.value)
                    break
        if assignment is not None:
            break
    return assignment


def get_func_rdr_model_path(func: Callable, model_dir: str) -> str:
    """
    :param func: The function to get the model path for.
    :param model_dir: The directory to save the model to.
    :return: The path to the model file.
    """
    func_name = get_method_name(func)
    func_class_name = get_method_class_name_if_exists(func)
    func_file_name = get_method_file_name(func)
    model_name = func_file_name
    model_name += f"_{func_class_name}" if func_class_name else ""
    model_name += f"_{func_name}"
    return os.path.join(model_dir, f"{model_name}.json")


def get_method_args_as_dict(method: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Get the arguments of a method as a dictionary.

    :param method: The method to get the arguments from.
    :param args: The positional arguments.
    :param kwargs: The keyword arguments.
    :return: A dictionary of the arguments.
    """
    func_arg_names = method.__code__.co_varnames
    func_arg_values = args + tuple(kwargs.values())
    return dict(zip(func_arg_names, func_arg_values))


def get_method_name(method: Callable) -> str:
    """
    Get the name of a method.

    :param method: The method to get the name of.
    :return: The name of the method.
    """
    return method.__name__ if hasattr(method, "__name__") else str(method)


def get_method_class_name_if_exists(method: Callable) -> Optional[str]:
    """
    Get the class name of a method if it has one.

    :param method: The method to get the class name of.
    :return: The class name of the method.
    """
    if hasattr(method, "__self__") and hasattr(method.__self__, "__class__"):
        return method.__self__.__class__.__name__
    return None


def get_method_file_name(method: Callable) -> str:
    """
    Get the file name of a method.

    :param method: The method to get the file name of.
    :return: The file name of the method.
    """
    return method.__code__.co_filename


def flatten_list(a: List):
    a_flattened = []
    for c in a:
        if is_iterable(c):
            a_flattened.extend(list(c))
        else:
            a_flattened.append(c)
    return a_flattened


def make_list(value: Any) -> List:
    """
    Make a list from a value.

    :param value: The value to make a list from.
    """
    return list(value) if is_iterable(value) else [value]


def is_iterable(obj: Any) -> bool:
    """
    Check if an object is iterable.

    :param obj: The object to check.
    """
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, type))


def get_type_from_string(type_path: str):
    """
    Get a type from a string describing its path using the format "module_path.ClassName".

    :param type_path: The path to the type.
    """
    module_path, class_name = type_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    if module == builtins and class_name == 'NoneType':
        return type(None)
    return getattr(module, class_name)


def get_full_class_name(cls):
    """
    Returns the full name of a class, including the module name.
    Copied from: https://github.com/tomsch420/random-events/blob/master/src/random_events/utils.py#L6C1-L21C101

    :param cls: The class.
    :return: The full name of the class
    """
    return cls.__module__ + "." + cls.__name__


def recursive_subclasses(cls):
    """
    Copied from: https://github.com/tomsch420/random-events/blob/master/src/random_events/utils.py#L6C1-L21C101
    :param cls: The class.
    :return: A list of the classes subclasses.
    """
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in recursive_subclasses(s)]


class SubclassJSONSerializer:
    """
    Originally from: https://github.com/tomsch420/random-events/blob/master/src/random_events/utils.py#L6C1-L21C101
    Class for automatic (de)serialization of subclasses.
    Classes that inherit from this class can be serialized and deserialized automatically by calling this classes
    'from_json' method.
    """

    def to_json_file(self, filename: str):
        """
        Save the object to a json file.
        """
        data = self.to_json()
        # save the json to a file
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        return data

    def to_json(self) -> Dict[str, Any]:
        return {"_type": get_full_class_name(self.__class__), **self._to_json()}

    def _to_json(self) -> Dict[str, Any]:
        """
        Create a json dict from the object.
        """
        raise NotImplementedError()

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create a variable from a json dict.
        This method is called from the from_json method after the correct subclass is determined and should be
        overwritten by the respective subclass.

        :param data: The json dict
        :return: The deserialized object
        """
        raise NotImplementedError()

    @classmethod
    def from_json_file(cls, filename: str):
        """
        Create an instance of the subclass from the data in the given json file.

        :param filename: The filename of the json file.
        """
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename, "r") as f:
            scrdr_json = json.load(f)
        return cls.from_json(scrdr_json)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create the correct instanceof the subclass from a json dict.

        :param data: The json dict
        :return: The correct instance of the subclass
        """
        if data is None:
            return None
        if isinstance(data, list):
            # if the data is a list, deserialize it
            return [cls.from_json(d) for d in data]
        elif isinstance(data, dict):
            if '__dataclass__' in data:
                # if the data is a dataclass, deserialize it
                return deserialize_dataclass(data)
            elif '_type' not in data:
                return {k: cls.from_json(v) for k, v in data.items()}
        elif not isinstance(data, dict):
            return data

        # check if type module is builtins
        data_type = get_type_from_string(data["_type"])
        if len(data) == 1:
            return data_type
        if data_type == NoneType:
            return None
        if data_type.__module__ == 'builtins':
            if is_iterable(data['value']) and not isinstance(data['value'], dict):
                return data_type([cls.from_json(d) for d in data['value']])
            return data_type(data["value"])
        if get_full_class_name(cls) == data["_type"]:
            data.pop("_type")
            return cls._from_json(data)
        for subclass in recursive_subclasses(SubclassJSONSerializer):
            if get_full_class_name(subclass) == data["_type"]:
                subclass_data = deepcopy(data)
                subclass_data.pop("_type")
                return subclass._from_json(subclass_data)

        raise ValueError("Unknown type {}".format(data["_type"]))

    save = to_json_file
    load = from_json_file


def copy_case(case: Union[Case, SQLTable]) -> Union[Case, SQLTable]:
    """
    Copy a case.

    :param case: The case to copy.
    :return: The copied case.
    """
    if isinstance(case, SQLTable):
        return copy_orm_instance_with_relationships(case)
    else:
        return deepcopy(case)


def copy_orm_instance(instance: SQLTable) -> SQLTable:
    """
    Copy an ORM instance by expunging it from the session then deep copying it and adding it back to the session. This
    is useful when you want to copy an instance and make changes to it without affecting the original instance.

    :param instance: The instance to copy.
    :return: The copied instance.
    """
    session: Session = inspect(instance).session
    session.expunge(instance)
    new_instance = deepcopy(instance)
    session.add(instance)
    return new_instance


def copy_orm_instance_with_relationships(instance: SQLTable) -> SQLTable:
    """
    Copy an ORM instance with its relationships (i.e. its foreign keys).

    :param instance: The instance to copy.
    :return: The copied instance.
    """
    instance_cp = copy_orm_instance(instance)
    for rel in class_mapper(instance.__class__).relationships:
        related_obj = getattr(instance, rel.key)
        if related_obj is not None:
            setattr(instance_cp, rel.key, related_obj)
    return instance_cp


def get_value_type_from_type_hint(attr_name: str, obj: Any) -> Type:
    """
    Get the value type from the type hint of an object attribute.

    :param attr_name: The name of the attribute.
    :param obj: The object to get the attributes from.
    """
    hint, origin, args = get_hint_for_attribute(attr_name, obj)
    if not origin and not hint:
        if hasattr(obj, attr_name):
            attr_value = getattr(obj, attr_name)
            if attr_value is not None:
                return type(attr_value)
        raise ValueError(f"Couldn't get type for Attribute {attr_name}, please provide a type hint")
    if origin in [list, set, tuple, type, dict]:
        attr_value_type = args[0]
    elif hint:
        attr_value_type = hint
    else:
        raise ValueError(f"Attribute {attr_name} has unsupported type {hint}.")
    return attr_value_type


def get_hint_for_attribute(attr_name: str, obj: Any) -> Tuple[Optional[Any], Optional[Any], Tuple[Any]]:
    """
    Get the type hint for an attribute of an object.

    :param attr_name: The name of the attribute.
    :param obj: The object to get the attribute from.
    :return: The type hint of the attribute.
    """
    if attr_name is None or not hasattr(obj.__class__, attr_name):
        return None, None, ()
    class_attr = getattr(obj.__class__, attr_name)
    if isinstance(class_attr, property):
        if not class_attr.fget:
            raise ValueError(f"Attribute {attr_name} has no getter.")
        hint = get_type_hints(class_attr.fget)['return']
    else:
        try:
            hint = get_type_hints(obj.__class__)[attr_name]
        except KeyError:
            hint = type(class_attr)
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is Mapped:
        return args[0], get_origin(args[0]), get_args(args[0])
    else:
        return hint, origin, args


def table_rows_as_str(row_dict: Dict[str, Any], columns_per_row: int = 9):
    """
    Print a table row.

    :param row_dict: The row to print.
    :param columns_per_row: The maximum number of columns per row.
    """
    all_items = list(row_dict.items())
    # make items a list of n rows such that each row has a max size of 4
    all_items = [all_items[i:i + columns_per_row] for i in range(0, len(all_items), columns_per_row)]
    keys = [list(map(lambda i: i[0], row)) for row in all_items]
    values = [list(map(lambda i: i[1], row)) for row in all_items]
    all_table_rows = []
    for row_keys, row_values in zip(keys, values):
        row_values = [str(v) if v is not None else "" for v in row_values]
        table = tabulate([row_values], headers=row_keys, tablefmt='plain', maxcolwidths=[20] * len(row_keys))
        all_table_rows.append(table)
    return "\n".join(all_table_rows)


def row_to_dict(obj):
    return {
        col.name: getattr(obj, col.name)
        for col in obj.__table__.columns
        if not col.primary_key and not col.foreign_keys
    }


def get_attribute_name(obj: Any, attribute: Optional[Any] = None, attribute_type: Optional[Type] = None,
                       possible_value: Optional[Any] = None) -> Optional[str]:
    """
    Get the name of an attribute from an object. The attribute can be given as a value, a type or a target value.
    And this method will try to find the attribute name using the given information.

    :param obj: The object to get the attribute name from.
    :param attribute: The attribute to get the name of.
    :param attribute_type: The type of the attribute to get the name of.
    :param possible_value: A possible value of the attribute to get the name of.
    :return: The name of the attribute.
    """
    attribute_name: Optional[str] = None
    if attribute_name is None and attribute is not None:
        attribute_name = get_attribute_name_from_value(obj, attribute)
    if attribute_name is None and attribute_type is not None:
        attribute_name = get_attribute_by_type(obj, attribute_type)[0]
    if attribute_name is None and possible_value is not None:
        attribute_name = get_attribute_by_type(obj, type(possible_value))[0]
    return attribute_name


def get_attribute_by_type(obj: Any, prop_type: Type) -> Tuple[Optional[str], Optional[Any]]:
    """
    Get a property from an object by type.

    :param obj: The object to get the property from.
    :param prop_type: The type of the property.
    """
    for name in dir(obj):
        if name.startswith("_") or callable(getattr(obj, name)):
            continue
        if isinstance(getattr(obj, name), (MetaData, registry)):
            continue
        prop_value = getattr(obj, name)
        if isinstance(prop_value, prop_type):
            return name, prop_value
        if hasattr(prop_value, "__iter__") and not isinstance(prop_value, str):
            if len(prop_value) > 0 and any(isinstance(v, prop_type) for v in prop_value):
                return name, prop_value
            else:
                # get args of type hint
                hint, origin, args = get_hint_for_attribute(name, obj)
                if origin in [list, set, tuple, dict, List, Set, Tuple, Dict]:
                    if prop_type is args[0]:
                        return name, prop_value
        else:
            # get the type hint of the attribute
            hint, origin, args = get_hint_for_attribute(name, obj)
            if hint is prop_type:
                return name, prop_value
            elif origin in [list, set, tuple, dict, List, Set, Tuple, Dict]:
                if prop_type is args[0]:
                    return name, prop_value
    return None, None


def get_attribute_name_from_value(obj: Any, attribute_value: Any) -> Optional[str]:
    """
    Get the name of an attribute from an object.

    :param obj: The object to get the attribute name from.
    :param attribute_value: The attribute value to get the name of.
    """
    for name in dir(obj):
        if name.startswith("_") or callable(getattr(obj, name)):
            continue
        prop_value = getattr(obj, name)
        if prop_value is attribute_value:
            return name


def get_attribute_values_transitively(obj: Any, attribute: Any) -> Any:
    """
    Get an attribute from a python object, if it is iterable, get the attribute values from all elements and unpack them
    into a list.

    :param obj: The object to get the sub attribute from.
    :param attribute: The  attribute to get.
    """
    if hasattr(obj, "__iter__") and not isinstance(obj, str):
        if isinstance(obj, (dict, UserDict)):
            all_values = [get_attribute_values_transitively(v, attribute) for v in obj.values()
                          if not isinstance(v, (str, type)) and hasattr(v, attribute)]
        else:
            all_values = [get_attribute_values_transitively(a, attribute) for a in obj
                          if not isinstance(a, (str, type)) and hasattr(a, attribute)]
        if can_be_a_set(all_values):
            return set().union(*all_values)
        else:
            return set(all_values)
    return getattr(obj, attribute)


def can_be_a_set(value: Any) -> bool:
    """
    Check if a value can be a set.

    :param value: The value to check.
    """
    if hasattr(value, "__iter__") and not isinstance(value, str):
        if len(value) > 0 and any(hasattr(v, "__iter__") and not isinstance(v, str) for v in value):
            return False
        else:
            return True
    else:
        return False


def get_all_subclasses(cls: Type) -> Dict[str, Type]:
    """
    Get all subclasses of a class recursively.

    :param cls: The class to get the subclasses of.
    :return: A dictionary of all subclasses.
    """
    all_subclasses: Dict[str, Type] = {}
    for sub_cls in cls.__subclasses__():
        all_subclasses[sub_cls.__name__.lower()] = sub_cls
        all_subclasses.update(get_all_subclasses(sub_cls))
    return all_subclasses


def make_set(value: Any) -> Set[Any]:
    """
    Make a set from a value.

    :param value: The value to make a set from.
    """
    return set(value) if is_iterable(value) else {value}


def make_value_or_raise_error(value: Any) -> Any:
    """
    Make a value or raise an error if the value is not a single value.

    :param value: The value to check.
    """
    if hasattr(value, "__iter__") and not isinstance(value, str):
        if hasattr(value, "__len__") and len(value) == 1:
            return list(value)[0]
        else:
            raise ValueError(f"Expected a single value, got {value}")
    return value


def tree_to_graph(root_node: Node) -> nx.DiGraph:
    """
    Convert anytree to a networkx graph.

    :param root_node: The root node of the tree.
    :return: A networkx graph.
    """
    graph = nx.DiGraph()
    unique_node_names = get_unique_node_names_func(root_node)

    def add_edges(node):
        if unique_node_names(node) not in graph.nodes:
            graph.add_node(unique_node_names(node))
        for child in node.children:
            if unique_node_names(child) not in graph.nodes:
                graph.add_node(unique_node_names(child))
            graph.add_edge(unique_node_names(node), unique_node_names(child), weight=child.weight)
            add_edges(child)

    add_edges(root_node)
    return graph


def get_unique_node_names_func(root_node) -> Callable[[Node], str]:
    nodes = [root_node]

    def get_all_nodes(node):
        for c in node.children:
            nodes.append(c)
            get_all_nodes(c)

    get_all_nodes(root_node)

    def nodenamefunc(node: Node):
        """
        Set the node name for the dot exporter.
        """
        similar_nodes = [n for n in nodes if n.name == node.name]
        node_idx = similar_nodes.index(node)
        return node.name if node_idx == 0 else f"{node.name}_{node_idx}"

    return nodenamefunc


def edge_attr_setter(parent, child):
    """
    Set the edge attributes for the dot exporter.
    """
    if child and hasattr(child, "weight") and child.weight:
        return f'style="bold", label=" {child.weight}"'
    return ""


def render_tree(root: Node, use_dot_exporter: bool = False,
                filename: str = "scrdr"):
    """
    Render the tree using the console and optionally export it to a dot file.

    :param root: The root node of the tree.
    :param use_dot_exporter: Whether to export the tree to a dot file.
    :param filename: The name of the file to export the tree to.
    """
    if not root:
        logging.warning("No rules to render")
        return
    for pre, _, node in RenderTree(root):
        print(f"{pre}{node.weight if hasattr(node, 'weight') and node.weight else ''} {node.__str__()}")
    if use_dot_exporter:
        unique_node_names = get_unique_node_names_func(root)

        de = DotExporter(root,
                         nodenamefunc=unique_node_names,
                         edgeattrfunc=edge_attr_setter
                         )
        de.to_dotfile(f"{filename}{'.dot'}")
        de.to_picture(f"{filename}{'.png'}")


def draw_tree(root: Node, fig: plt.Figure):
    """
    Draw the tree using matplotlib and networkx.
    """
    if root is None:
        return
    fig.clf()
    graph = tree_to_graph(root)
    fig_sz_x = 13
    fig_sz_y = 10
    fig.set_size_inches(fig_sz_x, fig_sz_y)
    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot")
    # scale down pos
    max_pos_x = max([v[0] for v in pos.values()])
    max_pos_y = max([v[1] for v in pos.values()])
    pos = {k: (v[0] * fig_sz_x / max_pos_x, v[1] * fig_sz_y / max_pos_y) for k, v in pos.items()}
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1000,
            ax=fig.gca(), node_shape="o", font_size=8)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'),
                                 ax=fig.gca(), rotate=False, clip_on=False)
    plt.pause(0.1)
