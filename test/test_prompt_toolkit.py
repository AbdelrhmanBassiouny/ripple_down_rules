from __future__ import annotations

import ast

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from typing_extensions import Set, Optional, List

from ripple_down_rules.datastructures import str_to_operator_fn
from ripple_down_rules.utils import make_set, get_attribute_values


class PhysicalObject:
    def __init__(self, name: str):
        self.name = name
        self._contained_objects: Set[PhysicalObject] = set()

    @property
    def contained_objects(self):
        return self._contained_objects

    @contained_objects.setter
    def contained_objects(self, value):
        self._contained_objects = value

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Part(PhysicalObject):
    ...


class Robot(PhysicalObject):
    def __init__(self, name: str, parts: Optional[List[Part]] = None):
        super().__init__(name)
        self.parts = parts or []


part_a = Part(name="A")
part_b = Part(name="B")
part_c = Part(name="C")
part_d = Part(name="D")
part_e = Part(name="E")
robot = Robot("pr2", parts=[part_a, part_b, part_c, part_d])
part_a.contained_objects = {part_b, part_c}
part_c.contained_objects = {part_d}
part_d.contained_objects = {part_e}
case = robot

# Define completer with all object attributes and comparison operators
complete = ['==', '!=', '>', '<', '>=', '<=']
complete += ["isinstance(", "issubclass(", "type(", "len(", "hasattr(", "getattr(", "setattr(", "delattr("]
case_attrs = [attr for attr in dir(case) if not attr.startswith("__")]
sub_attrs = {attr: [sub_attr for sub_attr in dir(getattr(case, attr)) if not sub_attr.startswith("__")] for attr in case_attrs}
case_attr_types = {attr: type(getattr(case, attr)) for attr in case_attrs}
sub_attrs_types = {attr: {sub_attr: type(getattr(getattr(case, attr), sub_attr)) for sub_attr in sub_attrs[attr]} for attr in case_attrs}


def add_completions(case, complete):
    for attr in dir(case):
        if attr.startswith("__") or attr.startswith("_"):
            continue
        complete.append(f"{case.__class__.__name__}.{attr}")
        for sub_attr in dir(getattr(case, attr)):
            if sub_attr.startswith("__"):
                continue
            if hasattr(sub_attr, "__iter__") and not isinstance(sub_attr, str):
                for sub_attr_element in sub_attr:
                    complete.append(f"{attr}.{sub_attr_element}")
                    complete.append(f"{case.__class__.__name__}.{attr}.{sub_attr_element}")
            else:
                complete.append(f"{attr}.{sub_attr}")
                complete.append(f"{case.__class__.__name__}.{attr}.{sub_attr}")
    return complete


add_completions(case, complete)
completer = WordCompleter(complete)
session = PromptSession(completer=completer)


while True:
    user_input = session.prompt(f"\nGive Conclusion on {case.__class__.__name__}.contained_objects >>> ")
    if user_input.lower() in ['exit', 'quit', '']:
        break
    print(f"Evaluating: {user_input}")
    try:
        # Parse the input into an AST
        tree = ast.parse(user_input, mode='eval')
        print(f"AST parsed successfully: {ast.dump(tree)}")
        attr_chain = user_input.split('.')
        user_attr = attr_chain[0]
        user_sub_attr = attr_chain[1] if len(attr_chain) > 1 else None
        # Evaluate expression
        if user_sub_attr:
            attr = getattr(case, user_attr)
            attr = get_attribute_values(attr, user_sub_attr)
        else:
            attr = getattr(case, user_attr)
        attr = set().union(*attr) if hasattr(attr, "__iter__") and not isinstance(attr, str) else attr
        print(f"Evaluated expression: {attr}")
    except SyntaxError as e:
        print(f"Syntax error: {e}")
