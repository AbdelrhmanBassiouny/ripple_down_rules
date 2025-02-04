from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod

from orderedset import OrderedSet
from typing_extensions import Optional, Dict, TYPE_CHECKING, List, Tuple

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
    all_expert_answers: Optional[List] = None
    """
    A list of all expert answers, used for testing purposes.
    """
    use_loaded_answers: bool = False
    """
    A flag to indicate if the expert should use loaded answers or not.
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

    @abstractmethod
    def ask_if_conclusion_is_correct(self, x: Case, conclusion: Category,
                                     target: Optional[Category] = None,
                                     current_conclusions: Optional[List[Category]] = None) -> bool:
        """
        Ask the expert if the conclusion is correct.

        :param x: The case to classify.
        :param conclusion: The conclusion to check.
        :param target: The target category to compare the case with.
        :param current_conclusions: The current conclusions for the case.
        """
        pass


class Human(Expert):
    """
    The Human Expert class, an expert that asks the human to provide differentiating features and conclusions.
    """
    def __init__(self, use_loaded_answers: bool = False):
        self.all_expert_answers = []
        self.use_loaded_answers = use_loaded_answers

    def save_answers(self, path: str):
        """
        Save the expert answers to a file.

        :param path: The path to save the answers to.
        """
        with open(path + '.json', "w") as f:
            json.dump(self.all_expert_answers, f)

    def load_answers(self, path: str):
        """
        Load the expert answers from a file.

        :param path: The path to load the answers from.
        """
        with open(path + '.json', "r") as f:
            self.all_expert_answers = json.load(f)

    def ask_for_conditions(self, x: Case, target: Category, last_evaluated_rule: Optional[Rule] = None)\
            -> Dict[str, Condition]:
        if last_evaluated_rule and not self.use_loaded_answers:
            action = "Refinement" if last_evaluated_rule.fired else "Alternative"
            print(f"{action} needed for rule:\n")
        if last_evaluated_rule and last_evaluated_rule.fired:
            all_attributes = last_evaluated_rule.corner_case.attributes_list + x.attributes_list
        else:
            if not self.use_loaded_answers:
                print("Please provide a rule for case:")
            all_attributes = x.attributes_list

        all_names, max_len = self.get_all_names_and_max_len(all_attributes)

        if not self.use_loaded_answers:
            self.print_all_names(all_names, max_len)

            if last_evaluated_rule and last_evaluated_rule.fired:
                last_evaluated_rule.corner_case.print_values(all_names, is_corner_case=True,
                                                             ljust_sz=max_len)
            x.print_values(all_names, target, ljust_sz=max_len)

        # take user input
        return self._get_conditions(all_names, conditions_for="differentiating features")

    def ask_for_extra_conclusions(self, x: Case, current_conclusions: List[Category]) \
            -> Dict[Category, Dict[str, Condition]]:
        all_names, max_len = self.get_all_names_and_max_len(x.attributes_list)
        if not self.use_loaded_answers:
            self.print_all_names(all_names, max_len)
            x.print_values(all_names, conclusions=current_conclusions, ljust_sz=max_len)
        extra_conclusions = {}
        while True:
            if not self.use_loaded_answers:
                print("Please provide the extra conclusion or press enter to end:")
            if self.use_loaded_answers:
                value = self.all_expert_answers.pop(0)
            else:
                value = input()
                self.all_expert_answers.append(value)
            if not value:
                break
            extra_conclusions[Category(value)] = self._get_conditions(all_names, conditions_for="extra conclusions")
        return extra_conclusions

    def ask_if_conclusion_is_correct(self, x: Case, conclusion: Category,
                                     target: Optional[Category] = None,
                                     current_conclusions: Optional[List[Category]] = None) -> bool:
        """
        Ask the expert if the conclusion is correct.

        :param x: The case to classify.
        :param conclusion: The conclusion to check.
        :param target: The target category to compare the case with.
        :param current_conclusions: The current conclusions for the case.
        """
        if not self.use_loaded_answers:
            print(f"Is the conclusion {conclusion.name} correct for the case (y/n):")
            x.conclusions = current_conclusions
            print(x)
        while True:
            if self.use_loaded_answers:
                answer = self.all_expert_answers.pop(0)
            else:
                answer = input()
                self.all_expert_answers.append(answer)
            if answer.lower() == "y":
                return True
            elif answer.lower() == "n":
                return False

    def _get_conditions(self, all_names: List[str], conditions_for: str = "") -> Dict[str, Condition]:
        """
        Get the conditions from the user.

        :param all_names: list of all attribute names.
        :return: the conditions as a dictionary.
        """
        if not self.use_loaded_answers:
            print(f"Please provide conditions for {conditions_for} as comma separated conditions using: <, >, <=, >=, ==:")
        while True:
            if self.use_loaded_answers:
                value = self.all_expert_answers.pop(0)
            else:
                value = input()
                self.all_expert_answers.append(value)
            rules = value.split(",")
            done = True
            messages = []
            rule_conditions = {}
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
            elif len(messages) > 0 and not self.use_loaded_answers:
                print("\n".join(messages))

    @staticmethod
    def get_all_names_and_max_len(all_attributes: List[Attribute]) -> Tuple[List[str], int]:
        """
        Get all attribute names and the maximum length of the names and values.

        :param all_attributes: list of attributes
        :return: list of names and the maximum length
        """
        all_names = list(OrderedSet([a.name for a in all_attributes]))
        max_len = max([len(name) for name in all_names])
        max_len = max(max_len, max([len(str(a.value)) for a in all_attributes])) + 4
        return all_names, max_len

    @staticmethod
    def print_all_names(all_names: List[str], max_len: int):
        """
        Print all attribute names.

        :param all_names: list of names
        :param max_len: maximum length
        """
        def ljust(s):
            return str(s).ljust(max_len)
        names_row = ljust(f"names: ")
        names_row += ljust("id")
        names_row += "".join([f"{ljust(name)}" for name in all_names + ["type"]])
        print(names_row)
