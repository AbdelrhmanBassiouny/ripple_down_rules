from __future__ import annotations

from abc import ABC, abstractmethod

from orderedset import OrderedSet
from typing_extensions import Optional, Dict, TYPE_CHECKING, List

from .datastructures import Attribute, str_to_operator_fn, Condition, Case, Category
from .failures import InvalidOperator

if TYPE_CHECKING:
    from .rdr import Rule


class Expert(ABC):
    """
    The Abstract Expert class, all experts should inherit from this class.
    An expert is a class that can provide differentiating features and conclusions for a case when asked.
    The expert can compare a case with a corner case and provide the differentiating features and can also
    provide one or multiple conclusions for a case.
    """

    @abstractmethod
    def ask_for_conditions(self, x: Case, target: Category, last_evaluated_rule: Optional[Rule] = None)\
            -> Dict[str, Condition]:
        """
        Ask the expert to provide the differentiating features between two cases or unique features for a case
        that doesn't have a corner case to compare to.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :param last_evaluated_rule: The last evaluated rule.
        :return: The differentiating features as new rule conditions.
        """
        pass

    @abstractmethod
    def ask_for_extra_conclusions(self, x: Case, current_conclusions: List[Category]) \
            -> Dict[Category, Dict[str, Condition]]:
        """
        Ask the expert to provide extra conclusions for a case by providing a pair of category and conditions for
        that category.

        :param x: The case to classify.
        :param current_conclusions: The current conclusions for the case.
        :return: The extra conclusions for the case.
        """
        pass


class Human(Expert):
    """
    The Human Expert class, an expert that asks the human to provide differentiating features and conclusions.
    """

    def ask_for_conditions(self, x: Case, target: Category, last_evaluated_rule: Optional[Rule] = None)\
            -> Dict[str, Condition]:
        if last_evaluated_rule:
            action = "Refinement" if last_evaluated_rule.fired else "Alternative"
            print(f"{action} needed for rule:\n")
        if last_evaluated_rule and last_evaluated_rule.fired:
            all_attributes = list(last_evaluated_rule.corner_case.attributes.values()) + list(x.attributes.values())
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

        if last_evaluated_rule and last_evaluated_rule.fired:
            last_evaluated_rule.corner_case.print_values(all_names, is_corner_case=True,
                                                         ljust_sz=max_len)
        x.print_values(all_names, target, ljust_sz=max_len)

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

    def ask_for_extra_conclusions(self, x: Case, current_conclusions: List[Category]) \
            -> Dict[Category, Dict[str, Condition]]:
        pass
