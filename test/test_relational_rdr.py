import os
from typing import Optional

from typing_extensions import Any

from relational_rdr_test_case import RelationalRDRTestCase
from ripple_down_rules.datastructures import RDRMode, ObjectAttributeTarget, Case
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR
from ripple_down_rules.utils import render_tree, prompt_for_relational_conditions


def test_classify_scrdr(obj: Any, target_property: Any,
                        target_value: Optional[Any] = None, expert_answers_dir="./test_expert_answers"):
    use_loaded_answers = False
    save_answers = False
    filename = expert_answers_dir + "/relational_scrdr_expert_answers_classify"
    expert = Human(use_loaded_answers=use_loaded_answers, mode=RDRMode.Relational)
    if use_loaded_answers:
        expert.load_answers(filename)

    scrdr = SingleClassRDR(mode=RDRMode.Relational)
    case = Case.from_object(obj)
    cat = scrdr.fit_case(case, for_attribute=target_property, expert=expert,
                         mode=RDRMode.Relational)
    render_tree(scrdr.start_rule, use_dot_exporter=True, filename="./test_results/relational_scrdr_classify")
    if target_value:
        assert cat == target_value

    if save_answers:
        cwd = os.getcwd()
        file = os.path.join(cwd, filename)
        expert.save_answers(file)


def test_parse_relational_conditions(case, target):
    user_input = "parts is not None and len(parts) > 0"
    target = RelationalRDRTestCase.target
    prompt_for_relational_conditions(case, target, user_input)


RelationalRDRTestCase.setUpClass()
robot = RelationalRDRTestCase.robot

# test_parse_relational_conditions(RelationalRDRTestCase.case, RelationalRDRTestCase.target)
test_classify_scrdr(robot, robot.contained_objects)
