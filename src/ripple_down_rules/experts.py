from __future__ import annotations

import json
from abc import ABC, abstractmethod

from typing_extensions import Optional, Dict, TYPE_CHECKING, List, Tuple, Type, Union

from .datastructures import str_to_operator_fn, Condition, Case
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
    def ask_for_conditions(self, x: Case, targets: List[Category], last_evaluated_rule: Optional[Rule] = None) \
            -> Dict[str, Condition]:
        """
        Ask the expert to provide the differentiating features between two cases or unique features for a case
        that doesn't have a corner case to compare to.

        :param x: The case to classify.
        :param targets: The target categories to compare the case with.
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
                                     targets: Optional[List[Category]] = None,
                                     current_conclusions: Optional[List[Category]] = None) -> bool:
        """
        Ask the expert if the conclusion is correct.

        :param x: The case to classify.
        :param conclusion: The conclusion to check.
        :param targets: The target categories to compare the case with.
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

    def ask_for_conditions(self, x: Case,
                           targets: Union[Category, List[Category]],
                           last_evaluated_rule: Optional[Rule] = None) \
            -> Dict[str, Condition]:
        targets = targets if isinstance(targets, list) else [targets]
        if last_evaluated_rule and not self.use_loaded_answers:
            action = "Refinement" if last_evaluated_rule.fired else "Alternative"
            print(f"{action} needed for rule:\n")
        if last_evaluated_rule and last_evaluated_rule.fired:
            all_attributes = last_evaluated_rule.corner_case.attributes_list + x.attributes_list
        else:
            if not self.use_loaded_answers:
                print("Please provide a rule for case:")
            all_attributes = x.attributes_list

        all_names, max_len = x.get_all_names_and_max_len(all_attributes)

        if not self.use_loaded_answers:
            x.print_all_names(all_names, max_len, target_types=list(map(type, targets)))

            if last_evaluated_rule and last_evaluated_rule.fired:
                last_evaluated_rule.corner_case.print_values(all_names, is_corner_case=True,
                                                             ljust_sz=max_len)
            x.print_values(all_names, targets=targets, ljust_sz=max_len)

        # take user input
        return self._get_conditions(all_names, conditions_for="differentiating features")

    def ask_for_extra_conclusions(self, x: Case, current_conclusions: List[Category]) \
            -> Dict[Category, Dict[str, Condition]]:
        """
        Ask the expert to provide extra conclusions for a case by providing a pair of category and conditions for
        that category.

        :param x: The case to classify.
        :param current_conclusions: The current conclusions for the case.
        :return: The extra conclusions for the case.
        """
        all_names, max_len = x.get_all_names_and_max_len()
        extra_conclusions = {}
        while True:
            category = self.ask_for_conclusion(x, current_conclusions)
            if not category:
                break
            extra_conclusions[category] = self._get_conditions(all_names, conditions_for="extra conclusions")
        return extra_conclusions

    def ask_for_conclusion(self, x: Case, current_conclusions: Optional[List[Category]] = None) -> Optional[Category]:
        """
        Ask the expert to provide a conclusion for the case.

        :param x: The case to classify.
        :param current_conclusions: The current conclusions for the case if any.
        """
        conclusion_types = list(map(type, current_conclusions)) if current_conclusions else None
        all_names, max_len = x.get_all_names_and_max_len()
        if not self.use_loaded_answers:
            x.print_all_names(all_names, max_len, conclusion_types=conclusion_types)
            x.print_values(all_names, conclusions=current_conclusions, ljust_sz=max_len)
        while True:
            if not self.use_loaded_answers:
                print("Please provide the conclusion as \"name:value\" or \"name\" or press enter to end:")
            if self.use_loaded_answers:
                value = self.all_expert_answers.pop(0)
            else:
                value = input()
                self.all_expert_answers.append(value)
            if value:
                try:
                    return self.parse_conclusion(value)
                except ValueError as e:
                    print(e)
            else:
                return None

    def parse_conclusion(self, value: str) -> Category:
        """
        Parse the conclusion from the user input. If the conclusion is not found in the known categories,
        a new category is created with the name and value else a new instance of the category is created with the value.

        :param value: The value to parse.
        :return: The parsed category name and value.
        :raises ValueError: If the category name contains non-alphabetic characters.
        """
        if ':' not in value:
            cat_name = "".join([w.capitalize() for w in value.split()])
            if not all(char.isalpha() for char in cat_name):
                raise ValueError(f"Category name {cat_name} should only contain alphabets")
            category = Category(cat_name)
        else:
            cat_name_value = value.split(":")
            cat_name = cat_name_value[0].strip(' "')
            if len(cat_name_value) == 2:
                cat_value = self.parse_value(cat_name_value[1])
                category = self.create_category_instance(cat_name, cat_value)
            else:
                raise ValueError(f"Input format \"{value}\" is not correct")
        return category

    def create_category_instance(self, cat_name: str, cat_value: str) -> Category:
        """
        Create a new category instance.

        :param cat_name: The name of the category.
        :param cat_value: The value of the category.
        :return: A new instance of the category.
        """
        category_type = self.get_category_type(cat_name)
        if not category_type:
            category_type = self.create_new_category_type(cat_name)
        return category_type(cat_value)

    @staticmethod
    def get_category_type(cat_name: str) -> Optional[Type[Category]]:
        """
        Get the category type from the known categories.

        :param cat_name: The name of the category.
        :return: The category type.
        """
        cat_name = cat_name.capitalize()
        known_categories = [c.__name__ for c in Category.__subclasses__()]
        category_type = None
        if cat_name in known_categories:
            category_type = Category.__subclasses__()[known_categories.index(cat_name)]
        return category_type

    def create_new_category_type(self, cat_name: str) -> Type[Category]:
        """
        Create a new category type.

        :param cat_name: The name of the category.
        :return: A new category type.
        """
        category_type: Type[Category] = type(cat_name, (Category,), {})
        if self.ask_if_category_is_mutually_exclusive(category_type):
            category_type.mutually_exclusive = True
        return category_type

    def ask_if_category_is_mutually_exclusive(self, category: Type[Category]) -> bool:
        """
        Ask the expert if the new category can have multiple values.

        :param category: The category to check.
        """
        question = f"Can a case have multiple values of the new category {category.__name__}? (y/n):"
        return self.ask_yes_no_question(question)

    def ask_if_conclusion_is_correct(self, x: Case, conclusion: Category,
                                     targets: Optional[List[Category]] = None,
                                     current_conclusions: Optional[List[Category]] = None) -> bool:
        """
        Ask the expert if the conclusion is correct.

        :param x: The case to classify.
        :param conclusion: The conclusion to check.
        :param targets: The target categories to compare the case with.
        :param current_conclusions: The current conclusions for the case.
        """
        question = ""
        if not self.use_loaded_answers:
            targets = targets or []
            targets = targets if isinstance(targets, list) else [targets]
            x.conclusions = current_conclusions
            x.targets = targets
            question = f"Is the conclusion {conclusion.value} correct for the case (y/n):" \
                       f"\n{str(x)}"
        return self.ask_yes_no_question(question)

    def ask_yes_no_question(self, question: str) -> bool:
        """
        Ask the expert a yes or no question.

        :param question: The question to ask.
        :return: The answer to the question.
        """
        if not self.use_loaded_answers:
            print(question)
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
            print(
                f"Please provide conditions for {conditions_for} as comma separated conditions using: <, >, <=, >=, ==:")
        while True:
            if self.use_loaded_answers:
                value = self.all_expert_answers.pop(0)
            else:
                value = input()
                self.all_expert_answers.append(value)
            rules = value.split(",")
            all_messages = []
            all_rule_conditions = {}
            for rule in rules:
                rule_conditions, messages = self.parse_rule(rule, all_names)
                all_messages += messages if messages else []
                all_rule_conditions.update(rule_conditions)
            if not all_messages:
                return all_rule_conditions
            elif not self.use_loaded_answers:
                print("\n".join(all_messages))

    def parse_rule(self, rule: str, all_names: List[str]) -> Tuple[Dict[str, Condition], List[str]]:
        """
        Parse the rule from the user input.

        :param rule: The rule to parse.
        :param all_names: list of all attribute names.
        :return: the rule conditions as a dictionary and the error messages.
        """
        rule_conditions = {}
        rule = rule.strip()
        messages, name, value, operator = self.validate_input_and_get_error_msgs(all_names, rule)
        if messages:
            messages.append(f"Please rewrite this condition: \"{rule}\"")
        else:
            # map value to string if it contains characters or quotes, else map to float
            parsed_value = self.parse_value(value)
            rule_conditions[name] = Condition(name, parsed_value, operator)
        return rule_conditions, messages

    def parse_value(self, value: str) -> Union[str, float, set]:
        """
        Parse the value from the user input.

        :param value: The value to parse.
        :return: The parsed value as a string, float or set of string or float.
        """
        if self.is_set(value):
            set_values = value[1:-1].split(",")
            for i, val in enumerate(set_values):
                set_values[i] = self.parse_string_or_float(val)
            parsed_value = set(set_values)
        else:
            parsed_value = self.parse_string_or_float(value)
        return parsed_value

    def parse_string_or_float(self, val: str) -> Union[str, float]:
        """
        Parse the value to a string or a float.

        :param val: The value to parse.
        :return: The parsed value as a string or a float.
        """
        if self.is_string(val):
            return val.strip(' "' + "'")
        elif self.is_float(val):
            return float(val)
        return val

    @staticmethod
    def is_float(val: str) -> bool:
        """
        Check if the value is a float.

        :param val: The value to check.
        """
        try:
            float(val)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_string(val: str) -> bool:
        """
        Check if the value is a string.

        :param val: The value to check.
        """
        return (val[0] in ["'", '"'] and val[0] == val[-1]) or any(char.isalpha() for char in val)

    @staticmethod
    def is_set(val: str) -> bool:
        """
        Check if the value is a set.

        :param val: The value to check.
        """
        return ((val[0] == "{" and val[-1] == "}")
                or (val[0] == "[" and val[-1] == "]")
                or (val[0] == "(" and val[-1] == ")"))

    @staticmethod
    def validate_input_and_get_error_msgs(all_names, rule) \
            -> Tuple[List[str], Optional[str], Optional[str], Optional[str]]:
        """
        Validate the input and get error messages.

        :param all_names: list of all attribute names.
        :param rule: The rule to validate.
        :return: list of error messages, and the name, value and operator of the rule.
        """
        try:
            name, value, operator = str_to_operator_fn(rule)
            messages = []
            if not name:
                messages.append(f"Name cannot be empty")
            elif name not in all_names:
                messages.append(f"Attribute {name} not found in the attributes")
            if not value:
                messages.append(f"Value seems to be wrong or missing")
            if not operator:
                messages.append(f"Operator seems to be wrong or missing")
            return messages, name, value, operator
        except InvalidOperator as e:
            messages = [str(e) + " please enter it again"]
            return messages, None, None, None
