from __future__ import annotations

import ast
import os

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from typing_extensions import Set, Optional, List

from ripple_down_rules.datastructures import str_to_operator_fn, RDRMode, Case
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR
from ripple_down_rules.utils import make_set, get_attribute_values, get_completions


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





# case = Case.from_object(robot)
case = robot
# target = type(case["contained_objects"])([part_b, part_c, part_d, part_e])
target = {part_b, part_c, part_d, part_e}

completions = get_completions(case)
completer = WordCompleter(completions)
session = PromptSession(completer=completer)


def test_classify_scrdr(case, target, expert_answers_dir="./test_expert_answers"):
    use_loaded_answers = False
    save_answers = True
    filename = expert_answers_dir + "/relational_scrdr_expert_answers_classify"
    expert = Human(use_loaded_answers=use_loaded_answers)
    if use_loaded_answers:
        expert.load_answers(filename)

    scrdr = SingleClassRDR(mode=RDRMode.Relational)
    cat = scrdr.fit_case(case, target, expert=expert)
    assert cat == target

    if save_answers:
        cwd = os.getcwd()
        file = os.path.join(cwd, filename)
        expert.save_answers(file)


test_classify_scrdr(case, target)
exit()
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
