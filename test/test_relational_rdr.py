import os
from typing import Optional

from typing_extensions import Any
from sqlalchemy.orm import DeclarativeBase as Table

from relational_rdr_test_case import RelationalRDRTestCase, Robot
from ripple_down_rules.datastructures import RDRMode, Case
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR
from ripple_down_rules.utils import render_tree, CallableExpression


def test_classify_scrdr(obj: Any, target_property: Any,
                        target_value: Optional[Any] = None, expert_answers_dir="./test_expert_answers"):
    use_loaded_answers = False
    save_answers = False
    filename = expert_answers_dir + "/relational_scrdr_expert_answers_classify"
    expert = Human(use_loaded_answers=use_loaded_answers, mode=RDRMode.Relational)
    if use_loaded_answers:
        expert.load_answers(filename)

    scrdr = SingleClassRDR(mode=RDRMode.Relational)
    case = Case.from_object(obj) if not isinstance(obj, (Case, Table)) else obj
    cat = scrdr.fit_case(case, for_attribute=target_property, expert=expert,
                         mode=RDRMode.Relational)
    render_tree(scrdr.start_rule, use_dot_exporter=True, filename="./test_results/relational_scrdr_classify")
    if target_value:
        assert cat == target_value

    if save_answers:
        cwd = os.getcwd()
        file = os.path.join(cwd, filename)
        expert.save_answers(file)


def test_parse_relational_conditions(case):
    user_input = "parts is not None and len(parts) > 0"
    conditions = CallableExpression(user_input, bool)
    print(conditions)
    print(conditions(case))
    assert conditions(case) == (case.parts is not None and len(case.parts) > 0)


RelationalRDRTestCase.setUpClass()
robot = RelationalRDRTestCase.robot
# test_parse_relational_conditions(RelationalRDRTestCase.case)
robot_without_parts = Robot("pr2")
case_without_parts = Case.from_object(robot_without_parts)
# test_parse_relational_conditions(case_without_parts)
test_classify_scrdr(robot, robot.contained_objects)
