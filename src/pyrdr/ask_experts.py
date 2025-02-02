from __future__ import annotations

from orderedset import OrderedSet
from typing_extensions import Optional, Dict, TYPE_CHECKING

from .datastructures import Attribute, str_to_operator_fn, Condition, Case, Category
from .failures import InvalidOperator

if TYPE_CHECKING:
    from .rdr import Rule


def ask_human(x: Case, target: Category, pred: Optional[Rule] = None,
              diff_attributes: Optional[Dict[str, Attribute]] = None) -> Dict[str, Condition]:
    """
    Ask the human to provide the differentiating features between two cases.

    :param x: The case to classify.
    :param target: The target category to compare the case with.
    :param pred: The predicted rule.
    :param diff_attributes: The differentiating attributes between the predicted rule and the case.
    :return: The differentiating features as new rule conditions.
    """
    if pred:
        action = "Refinement" if pred.fired else "Alternative"
        print(f"{action} needed for rule:\n")
    if pred and pred.fired:
        all_attributes = list(pred.corner_case.attributes.values()) + list(x.attributes.values())
    else:
        print("Please provide a rule for case:")
        all_attributes = list(x.attributes.values())

    all_names = list(OrderedSet([a.name for a in all_attributes]))
    max_len = max([len(name) for name in all_names])
    max_len = max(max_len, max([len(str(a.value)) for a in all_attributes])) + 4

    def ljust(s):
        return str(s).ljust(max_len)

    names_row = ljust(f"names: ")
    names_row += ljust("id")
    names_row += "".join([f"{ljust(name)}" for name in all_names + ["type"]])
    print(names_row)

    if pred and pred.fired:
        pred.corner_case.print_values(all_names, is_corner_case=True,
                                      ljust_sz=max_len)
    x.print_values(all_names, target, ljust_sz=max_len)

    if pred and pred.fired and diff_attributes:
        diff = {name: "Y" if name in diff_attributes else "N" for name in all_names}
        diff_row = ljust(f"diff: ") + ljust(" ")
        diff_row += "".join([f"{ljust(diff[name])}" for name in all_names])
        print(diff_row)

    # take user input
    rule_conditions = {}
    print(f"Please provide the differentiating features as comma separated conditions using: <, >, <=, >=, ==:")
    while True:
        value = input()
        rules = value.split(",")
        done = True
        messages = []
        for rule in rules:
            rule = rule.strip()
            try:
                name, value, operator = str_to_operator_fn(rule)
                if name and value and operator:
                    if name not in all_names:
                        messages.append(f"Attribute {name} not found in the attributes list please enter it again")
                        done = False
                        continue
                    rule_conditions[name] = Condition(name, float(value), operator)
            except InvalidOperator as e:
                messages.append(str(e) + " please enter it again")
                done = False
        if done:
            return rule_conditions
        elif len(messages) > 0:
            print("\n".join(messages))
