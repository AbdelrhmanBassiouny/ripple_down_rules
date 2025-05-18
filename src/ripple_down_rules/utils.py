from __future__ import annotations

import ast
import builtins
import copyreg
import importlib
import json
import logging
import os
import re
import threading
import uuid
from collections import UserDict
from copy import deepcopy, copy
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

import ast


def str_to_snake_case(snake_str: str) -> str:
    """
    Convert a string to snake case.

    :param snake_str: The string to convert.
    :return: The converted string.
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', snake_str)
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    # remove redundant underscores
    s1 = re.sub(r'_{2,}', '_', s1)
    # remove leading and trailing underscores
    s1 = re.sub(r'^_|_$', '', s1)
    return s1


def are_results_subclass_of_types(result_types: List[Any], types_: List[Type]) -> bool:
    """
    Check if all results are subclasses of the given types.

    :param result_types: The list of result types to check.
    :param types_: The list of types to check against.
    :return: True if all results are subclasses of the given types, False otherwise.
    """
    for rt in result_types:
        if not any(issubclass(rt, t) for t in types_):
            return False
    return True


def get_imports_from_types(types: List[Type]) -> List[str]:
    """
    Get the import statements for a list of types.

    :param types: The types to get the import statements for.
    :return: The import statements as a string.
    """
    imports = map(get_import_from_type, types)
    return list({i for i in imports if i is not None})


def get_import_from_type(type_: Type) -> Optional[str]:
    """
    Get the import statement for a given type.

    :param type_: The type to get the import statement for.
    :return: The import statement as a string.
    """
    if hasattr(type_, "__module__") and hasattr(type_, "__name__"):
        if type_.__module__ == "builtins":
            return
        return f"from {type_.__module__} import {type_.__name__}"


def get_imports_from_scope(scope: Dict[str, Any]) -> List[str]:
    """
    Get the imports from the given scope.

    :param scope: The scope to get the imports from.
    :return: The imports as a string.
    """
    imports = []
    for k, v in scope.items():
        if not hasattr(v, "__module__") or not hasattr(v, "__name__"):
            continue
        imports.append(f"from {v.__module__} import {v.__name__}")
    return imports


def extract_imports(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read(), filename=file_path)

    scope = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                asname = alias.asname or alias.name
                try:
                    scope[asname] = importlib.import_module(module_name)
                except ImportError as e:
                    print(f"Could not import {module_name}: {e}")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            for alias in node.names:
                name = alias.name
                asname = alias.asname or name
                try:
                    module = importlib.import_module(module_name)
                    scope[asname] = getattr(module, name)
                except (ImportError, AttributeError) as e:
                    print(f"Could not import {name} from {module_name}: {e}")

    return scope


def extract_function_source(file_path: str,
                            function_names: List[str], join_lines: bool = True,
                            return_line_numbers: bool = False,
                            include_signature: bool = True) \
        -> Union[Dict[str, Union[str, List[str]]],
        Tuple[Dict[str, Union[str, List[str]]], List[Tuple[int, int]]]]:
    """
    Extract the source code of a function from a file.

    :param file_path: The path to the file.
    :param function_names: The names of the functions to extract.
    :param join_lines: Whether to join the lines of the function.
    :param return_line_numbers: Whether to return the line numbers of the function.
    :param include_signature: Whether to include the function signature in the source code.
    :return: A dictionary mapping function names to their source code as a string if join_lines is True,
     otherwise as a list of strings.
    """
    with open(file_path, "r") as f:
        source = f.read()

    # Parse the source code into an AST
    tree = ast.parse(source)
    function_names = make_list(function_names)
    functions_source: Dict[str, Union[str, List[str]]] = {}
    line_numbers = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            # Get the line numbers of the function
            lines = source.splitlines()
            func_lines = lines[node.lineno - 1:node.end_lineno]
            if not include_signature:
                func_lines = func_lines[1:]
            line_numbers.append((node.lineno, node.end_lineno))
            functions_source[node.name] = "\n".join(func_lines) if join_lines else func_lines
            if len(functions_source) == len(function_names):
                break
    if len(functions_source) != len(function_names):
        raise ValueError(f"Could not find all functions in {file_path}: {function_names} not found,"
                         f"functions not found: {set(function_names) - set(functions_source.keys())}")
    if return_line_numbers:
        return functions_source, line_numbers
    return functions_source


def encapsulate_user_input(user_input: str, func_signature: str, func_doc: Optional[str] = None) -> str:
    """
    Encapsulate the user input string with a function definition.

    :param user_input: The user input string.
    :param func_signature: The function signature to use for encapsulation.
    :param func_doc: The function docstring to use for encapsulation.
    :return: The encapsulated user input string.
    """
    if func_signature not in user_input:
        new_user_input = func_signature + "\n    "
        if func_doc is not None:
            new_user_input += f"\"\"\"{func_doc}\"\"\"" + "\n    "
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
            # check if the statement is a function, if so then all its lines not just the line in line_map are needed.
            if isinstance(stmt, ast.FunctionDef):
                start_code_line = line_map[id(stmt)]
                end_code_line = start_code_line + stmt.end_lineno
                required_lines.extend(code_lines[start_code_line:end_code_line])
            else:
                required_lines.insert(0, code_lines[line_map[id(stmt)]])

    required_lines.append(code_lines[-1])  # Always include return
    return required_lines


def serialize_dataclass(obj: Any, seen=None) -> Any:
    """
    Recursively serialize a dataclass to a dictionary. If the dataclass contains any nested dataclasses, they will be
    serialized as well. If the object is not a dataclass, it will be returned as is.

    :param obj: The dataclass to serialize.
    :return: The serialized dataclass as a dictionary or the object itself if it is not a dataclass.
    """
    if seen is None:
        seen = {}

    obj_id = id(obj)
    if obj_id in seen:
        return {'$ref': seen[obj_id]}

    if is_dataclass(obj):
        uid = str(uuid.uuid4())
        seen[obj_id] = uid
        result = {
            '$id': uid,
            "__dataclass__": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
            'fields': {}
        }
        for f in fields(obj):
            value = getattr(obj, f.name)
            result['fields'][f.name] = serialize_dataclass(value, seen)
        return result
    elif isinstance(obj, list):
        return [serialize_dataclass(v, seen) for v in obj]
    elif isinstance(obj, dict):
        return {k: serialize_dataclass(v, seen) for k, v in obj.items()}
    else:
        try:
            json.dumps(obj)  # Check if the object is JSON serializable
            return obj
        except TypeError:
            return None


def deserialize_dataclass(data: Any, refs: Optional[Dict[str, Any]] = None) -> Any:
    refs = {} if refs is None else refs
    preloaded = preload_serialized_objects(data, refs)
    return resolve_refs(preloaded, refs)


def preload_serialized_objects(data: Any, refs: Dict[str, Any] = None) -> Any:
    """
    Recursively deserialize a dataclass from a dictionary, if the dictionary contains a key "__dataclass__" (Most likely
    created by the serialize_dataclass function), it will be treated as a dataclass and deserialized accordingly,
    otherwise it will be returned as is.

    :param data: The dictionary to deserialize.
    :return: The deserialized dataclass.
    """
    if refs is None:
        refs = {}

    if isinstance(data, dict):

        if '$ref' in data:
            ref_id = data['$ref']
            if ref_id not in refs:
                return {'$ref': data['$ref']}
            return refs[ref_id]

        elif '$id' in data and '__dataclass__' in data and 'fields' in data:
            cls_path = data['__dataclass__']
            module_name, class_name = cls_path.rsplit('.', 1)
            cls = getattr(importlib.import_module(module_name), class_name)

            dummy_instance = cls.__new__(cls)  # Don't call __init__ yet
            refs[data['$id']] = dummy_instance

            for f in fields(cls):
                raw_value = data['fields'].get(f.name)
                value = preload_serialized_objects(raw_value, refs)
                setattr(dummy_instance, f.name, value)

            return dummy_instance

        else:
            return {k: preload_serialized_objects(v, refs) for k, v in data.items()}

    elif isinstance(data, list):
        return [preload_serialized_objects(item, refs) for item in data]
    elif isinstance(data, dict):
        return {k: preload_serialized_objects(v, refs) for k, v in data.items()}

    return data  # Primitive


def resolve_refs(obj, refs, seen=None):
    if seen is None:
        seen = {}

    obj_id = id(obj)
    if obj_id in seen:
        return seen[obj_id]

    # Resolve if dict with $ref
    if isinstance(obj, dict) and '$ref' in obj:
        ref_id = obj['$ref']
        if ref_id not in refs:
            raise KeyError(f"$ref to unknown ID: {ref_id}")
        return refs[ref_id]

    elif is_dataclass(obj):
        seen[obj_id] = obj  # Mark before diving deeper
        for f in fields(obj):
            val = getattr(obj, f.name)
            resolved = resolve_refs(val, refs, seen)
            setattr(obj, f.name, resolved)
        return obj

    elif isinstance(obj, list):
        resolved_list = []
        seen[obj_id] = resolved_list
        for item in obj:
            resolved_list.append(resolve_refs(item, refs, seen))
        return resolved_list

    elif isinstance(obj, dict):
        resolved_dict = {}
        seen[obj_id] = resolved_dict
        for k, v in obj.items():
            resolved_dict[k] = resolve_refs(v, refs, seen)
        return resolved_dict

    return obj  # Primitive


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
    return os.path.join(model_dir, f"{get_func_rdr_model_name(func)}.json")


def get_func_rdr_model_name(func: Callable, include_file_name: bool = False) -> str:
    """
    :param func: The function to get the model name for.
    :return: The name of the model.
    """
    func_name = get_method_name(func)
    func_class_name = get_method_class_name_if_exists(func)
    if include_file_name:
        func_file_name = get_method_file_name(func).split(os.sep)[-1].split('.')[0]
        model_name = func_file_name + '_'
    else:
        model_name = ''
    model_name += f"{func_class_name}_" if func_class_name else ""
    model_name += f"{func_name}"
    return model_name


def extract_bracket_arguments(val: str) -> List[str]:
    """
    Extract arguments inside brackets into a list.

    :param val: The string containing brackets.
    :return: List of arguments inside brackets.
    """
    if '[' not in val:
        return [val]
    args_start = val.find('[')
    args_end = val.rfind(']')
    if args_end == -1:
        return [val]
    base_type = val[:args_start]
    args = val[args_start + 1:args_end].split(',')
    args = [arg.strip() for arg in args]
    return [base_type] + args


def typing_hint_to_str(type_hint: Any) -> Tuple[str, List[str]]:
    """
    Convert a typing hint to a string.

    :param type_hint: The typing hint to convert.
    :return: The string representation of the typing hint.
    """
    val = (str(type_hint).strip("<>")
           .replace("class ", "")
           # .replace("typing.", "")
           .replace("'", ""))
    all_args = []
    if '[' in val:
        args = extract_bracket_arguments(val)
        args_with_brackets = [arg for arg in args if '[' in arg]
        all_args.extend([arg for arg in args if '[' not in arg])
        while args_with_brackets:
            for arg in args:
                if '[' in arg:
                    sub_args = extract_bracket_arguments(arg)
                    args_with_brackets.remove(arg)
                    all_args.extend([sarg for sarg in sub_args if '[' not in sarg])
                    args_with_brackets.extend([sarg for sarg in sub_args if '[' in sarg])
                elif arg not in all_args:
                    all_args.append(arg)
            args = args_with_brackets
        for arg in all_args:
            val = val.replace(arg, arg.split('.')[-1])
    else:
        val = val.split('.')[-1]
    return val, all_args


def get_method_args_as_dict(method: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Get the arguments of a method as a dictionary.

    :param method: The method to get the arguments from.
    :param args: The positional arguments.
    :param kwargs: The keyword arguments.
    :return: A dictionary of the arguments.
    """
    func_arg_names = method.__code__.co_varnames
    func_arg_names = list(map(lambda arg_name: f"{arg_name}_" if arg_name in ["self", "cls"] else arg_name,
                              func_arg_names))
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
    if hasattr(method, "__self__"):
        if hasattr(method.__self__, "__class__"):
            return method.__self__.__class__.__name__
    return method.__qualname__.split('.')[0] if hasattr(method, "__qualname__") else None


def get_method_class_if_exists(method: Callable, *args) -> Optional[Type]:
    """
    Get the class of a method if it has one.

    :param method: The method to get the class of.
    :return: The class of the method, if it exists otherwise None.
    """
    if hasattr(method, "__self__"):
        if hasattr(method.__self__, "__class__"):
            return method.__self__.__class__
    elif method.__code__.co_varnames:
        if method.__code__.co_varnames[0] == 'self':
            return args[0].__class__
        elif method.__code__.co_varnames[0] == 'cls':
            return args[0]
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
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, type, bytes, bytearray))


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
    data_class_refs = {}

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

    @staticmethod
    def to_json_static(obj) -> Dict[str, Any]:
        if is_dataclass(obj):
            return serialize_dataclass(obj)
        return {"_type": get_full_class_name(obj.__class__), **obj._to_json()}

    def to_json(self) -> Dict[str, Any]:
        return self.to_json_static(self)

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
    def from_json_file(cls, filename: str) -> Any:
        """
        Create an instance of the subclass from the data in the given json file.

        :param filename: The filename of the json file.
        """
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename, "r") as f:
            scrdr_json = json.load(f)
        deserialized_obj = cls.from_json(scrdr_json)
        cls.data_class_refs.clear()
        return deserialized_obj

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
                return deserialize_dataclass(data, cls.data_class_refs)
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


def _pickle_thread(thread_obj) -> Any:
    """Return a plain object with user-defined attributes but no thread behavior."""

    class DummyThread:
        pass

    dummy = DummyThread()
    # Copy only non-thread-related attributes
    for attr, value in thread_obj.__dict__.items():
        print(attr)
        if not attr.startswith("_"):  # Skip internal Thread attributes
            setattr(dummy, attr, value)
    return dummy


copyreg.pickle(threading.Thread, _pickle_thread)


def copy_case(case: Union[Case, SQLTable]) -> Union[Case, SQLTable, Any]:
    """
    Copy a case.

    :param case: The case to copy.
    :return: The copied case.
    """
    if isinstance(case, SQLTable):
        return copy_orm_instance_with_relationships(case)
    else:
        # copy the case recursively for 1 level
        # try:
        #     case_copy = deepcopy(case)
        # except Exception as e:
        case_copy = copy(case)
        for attr in dir(case):
            if attr.startswith("_") or callable(getattr(case, attr)):
                continue
            attr_value = getattr(case, attr)
            if is_iterable(attr_value):
                setattr(case_copy, attr, copy(attr_value))
        return case_copy


def copy_orm_instance(instance: SQLTable) -> SQLTable:
    """
    Copy an ORM instance by expunging it from the session then deep copying it and adding it back to the session. This
    is useful when you want to copy an instance and make changes to it without affecting the original instance.

    :param instance: The instance to copy.
    :return: The copied instance.
    """
    session: Session = inspect(instance).session
    if session is not None:
        session.expunge(instance)
        new_instance = deepcopy(instance)
        session.add(instance)
    else:
        new_instance = instance
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


def get_hint_for_attribute(attr_name: str, obj: Any) -> Tuple[Optional[Type], Optional[Type], Tuple[Type]]:
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
    origin, args = get_origin_and_args_from_type_hint(hint)
    return origin, origin, args


def get_origin_and_args_from_type_hint(type_hint: Type) -> Tuple[Optional[Type], Tuple[Type]]:
    """
    Get the origin and arguments from a type hint.W

    :param type_hint: The type hint to get the origin and arguments from.
    :return: The origin and arguments of the type hint.
    """
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if origin is Mapped:
        return get_origin(args[0]), get_args(args[0])
    else:
        return origin, args


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
        row_values = [v.lower() if v in ["True", "False"] else v for v in row_values]
        table = tabulate([row_values], headers=row_keys, tablefmt='plain', maxcolwidths=[20] * len(row_keys))
        all_table_rows.append(table)
    return "\n".join(all_table_rows)


def row_to_dict(obj):
    return {
        col.name: getattr(obj, col.name)
        for col in obj.__table__.columns
        if not col.primary_key and not col.foreign_keys
    }


def dataclass_to_dict(obj):
    """
    Convert a dataclass to a dictionary.

    :param obj: The dataclass to convert.
    :return: The dictionary representation of the dataclass.
    """
    if is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in fields(obj) if not f.name.startswith("_")}
    else:
        raise ValueError(f"Object {obj} is not a dataclass.")


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


def make_set(value: Any) -> Set:
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
    if matplotlib.get_backend().lower() not in ['qt5agg', 'qt4agg', 'qt6agg']:
        matplotlib.use("Qt5Agg")  # or "Qt5Agg", depending on availability

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
