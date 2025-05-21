# Updated version: recursively trace where dependencies (like `sensors`) are defined too
import ast
import inspect
import json
import os
import sys
from textwrap import dedent

from ripple_down_rules.utils import extract_dependencies, extract_function_or_class_source


def find_var_definition_lines(var_name, frame, current_line=None):
    """Try to find the source code line where a variable was defined."""
    filename = frame.f_code.co_filename
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None, current_line, []

    current_line = frame.f_lineno - 1 if current_line is None else current_line
    for i in range(current_line - 1, -1, -1):
        if var_name in lines[i]:
            if '=' in lines[i]:
                src = lines[i].strip()
                return src, i, lines
            if 'def ' in lines[i]:
                # If we hit a function definition inclide ,the whole function
                # as a dependency
                func_name = lines[i].split('def ')[1].split('(')[0]
                func_src = extract_function_or_class_source(filename, [func_name], return_line_numbers=False,
                                                            join_lines=True, until_line_number=i)
                return func_src, i, lines
            if 'class ' in lines[i]:
                # If we hit a class definition, include the whole class
                class_name = lines[i].split('class ')[1].split(':')[0].split('(')[0]
                class_src = extract_function_or_class_source(filename, [class_name], return_line_numbers=False,
                                                             join_lines=True, until_line_number=i)
                return class_src, i, lines
            if 'import ' in lines[i]:
                # If we hit an import statement, include the whole import
                import_src = lines[i].strip()
                # Check if the import statement is a multi-line import
                if '(' in import_src:
                    # Find the closing parenthesis
                    for j in range(i, len(lines)):
                        if ')' in lines[j]:
                            if j > i:
                                # Include the whole import statement
                                import_src += ''.join(lines[i:j + 1])
                            break
                return import_src, i, lines
    return None, current_line, lines


def resolve_all_dependencies(var_name, frame, seen=None, current_line=None):
    """Recursively resolve dependencies for a variable."""
    if seen is None:
        seen = set()

    if var_name in seen:
        return []
    seen.add(var_name)

    src_line, lineno, lines = find_var_definition_lines(var_name, frame, current_line)
    if not src_line:
        return []

    deps = [(var_name, src_line, lineno)]

    try:
        parsed = ast.parse(src_line, mode="exec")
        list_com_targets = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.ListComp):
                for gen in node.generators:
                    list_com_targets.append(gen.target.id)
            if isinstance(node, ast.Name) and node.id in list_com_targets + [var_name]:
                continue
            deps += extract_dependencies_recursive(node, frame, seen, current_line=lineno)
    except Exception as e:
        raise RuntimeError(f"Error parsing line {lineno}: {src_line}") from e

    return deps


def extract_dependencies_recursive(node, frame, seen=None, current_line=None):
    """Enhanced: Recursively walk constructor AST and resolve all var definitions."""
    if seen is None:
        seen = set()

    deps = []

    if isinstance(node, ast.Call):
        deps += extract_dependencies_recursive(node.func, frame, seen, current_line)
        for arg in node.args:
            deps += extract_dependencies_recursive(arg, frame, seen, current_line)
        for kw in node.keywords:
            deps += extract_dependencies_recursive(kw.value, frame, seen, current_line)

    elif isinstance(node, ast.Name):
        deps += resolve_all_dependencies(node.id, frame, seen, current_line)

    elif isinstance(node, ast.ListComp):
        for gen in node.generators:
            deps += extract_dependencies_recursive(gen.iter, frame, seen, current_line)
            deps += extract_dependencies_recursive(gen.target, frame, seen, current_line)
        deps += extract_dependencies_recursive(node.elt, frame, seen, current_line)

    elif isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            deps += extract_dependencies_recursive(elt, frame, seen, current_line)

    return deps


def register_corner_case_with_trace_full(obj_expr_str):
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)

    try:
        expr_ast = ast.parse(obj_expr_str, mode='eval')
    except SyntaxError:
        return {"error": "Invalid expression"}

    deps = extract_dependencies_recursive(expr_ast.body, frame)
    # Deduplicate
    seen_keys = set()
    unique_deps = []
    for dep in deps:
        if dep[0] not in seen_keys:
            unique_deps.append(dep)
            seen_keys.add(dep[0])

    return {
        "constructed_expr": obj_expr_str,
        "dependencies": unique_deps,
        "filename": info.filename,
        "callsite_line": info.lineno,
        "callsite_code": info.code_context[0].strip() if info.code_context else None
    }


# Rerun test case
def test_case_with_nested_trace():
    sensors = ['cam', 'lidar']
    config = {"speed": 1.0}

    class RobotArm:
        def __init__(self, name): self.name = name

    class Sensor:
        def __init__(self, id): self.id = id

    class MyCase:
        def __init__(self, cfg, arm, sensors): self.cfg, self.arm, self.sensors = cfg, arm, sensors

    arm = RobotArm("UR5")
    inputs = [Sensor(x) for x in sensors]
    case = MyCase(config, arm, inputs)
    # return case
    trace_info = register_corner_case_with_trace_full("case")
    json.dump(trace_info, sys.stdout, indent=4)
    return trace_info

def test_extract_dependencies():
    func_source, line_numbers = extract_function_or_class_source(__file__,
                                                        "test_case_with_nested_trace",
                                                                 return_line_numbers=True,
                                                                 join_lines=True, until_line_number=132,
                                                                 include_signature=False)
    func_source = dedent(func_source["test_case_with_nested_trace"]).split(os.linesep)
    deps = extract_dependencies(func_source)
    print("Dependencies:", deps)
