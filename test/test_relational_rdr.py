import os

from relational_rdr_test_case import RelationalRDRTestCase
from ripple_down_rules.datastructures import RDRMode
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR


def test_classify_scrdr(case, target, expert_answers_dir="./test_expert_answers"):
    use_loaded_answers = False
    save_answers = True
    filename = expert_answers_dir + "/relational_scrdr_expert_answers_classify"
    expert = Human(use_loaded_answers=use_loaded_answers, mode=RDRMode.Relational)
    if use_loaded_answers:
        expert.load_answers(filename)

    scrdr = SingleClassRDR(mode=RDRMode.Relational)
    cat = scrdr.fit_case(case, for_property=case._obj.contained_objects, target=target, expert=expert)
    assert cat == target

    if save_answers:
        cwd = os.getcwd()
        file = os.path.join(cwd, filename)
        expert.save_answers(file)


def test_parse_relational_conditions(case, target):
    user_input = "parts is not None and len(parts) > 0"
    Human.prompt_for_relational_conditions(case, [target], user_input)


RelationalRDRTestCase.setUpClass()

# test_parse_relational_conditions(RelationalRDRTestCase.case, RelationalRDRTestCase.target)
test_classify_scrdr(RelationalRDRTestCase.case, RelationalRDRTestCase.target)
