from __future__ import annotations

import ast
import json
from abc import ABC, abstractmethod

from typing_extensions import Optional, Dict, TYPE_CHECKING, List, Tuple, Type, Union, Any, Sequence, Callable

from .datastructures import Operator, Condition, Attribute, Case, RDRMode, Categorical, ObjectPropertyTarget
from .failures import InvalidOperator
from .utils import get_all_subclasses, get_property_name, VariableVisitor, \
    get_prompt_session_for_obj, parse_relational_conclusion, prompt_user_for_input, get_attribute_values

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
    known_categories: Optional[Dict[str, Type[Attribute]]] = None
    """
    The known categories (i.e. Attribute types) to use.
    """

    def __init__(self, mode: RDRMode = RDRMode.Propositional):
        self.mode: RDRMode = mode

    @abstractmethod
    def ask_for_conditions(self, x: Case, targets: List[Attribute], last_evaluated_rule: Optional[Rule] = None) \
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
    def ask_for_extra_conclusions(self, x: Case, current_conclusions: List[Attribute]) \
            -> Dict[Attribute, Dict[str, Condition]]:
        """
        Ask the expert to provide extra conclusions for a case by providing a pair of category and conditions for
        that category.

        :param x: The case to classify.
        :param current_conclusions: The current conclusions for the case.
        :return: The extra conclusions for the case.
        """
        pass

    @abstractmethod
    def ask_if_conclusion_is_correct(self, x: Case, conclusion: Attribute,
                                     targets: Optional[List[Attribute]] = None,
                                     current_conclusions: Optional[List[Attribute]] = None) -> bool:
        """
        Ask the expert if the conclusion is correct.

        :param x: The case to classify.
        :param conclusion: The conclusion to check.
        :param targets: The target categories to compare the case with.
        :param current_conclusions: The current conclusions for the case.
        """
        pass

    def ask_for_relational_conclusion(self, x: Case, for_attribute: Union[str, Attribute, Sequence[Attribute]]) \
            -> Optional[Attribute]:
        """
        Ask the expert to provide a relational conclusion for the case.

        :param x: The case to classify.
        :param for_attribute: The attribute to provide the conclusion for.
        """


class Human(Expert):
    """
    The Human Expert class, an expert that asks the human to provide differentiating features and conclusions.
    """

    def __init__(self, use_loaded_answers: bool = False, mode: RDRMode = RDRMode.Propositional):
        super().__init__(mode)
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

    def _get_relational_conditions(self, x: Case, targets: List[Attribute],
                                   conditions_for="differentiating features") -> Dict[str, Condition]:
        """
        Ask the expert to provide the differentiating features between two cases or unique features for a case
        that doesn't have a corner case to compare to.

        :param x: The case to classify.
        :param targets: The target categories to compare the case with.
        :param conditions_for: A string indicating what the conditions are for.
        :return: The differentiating features as new rule conditions.
        """
        while True:
            user_input = None
            if self.use_loaded_answers:
                user_input = self.all_expert_answers.pop(0)
            # Initialze prompt session and get conditions from user written input
            return self.prompt_for_relational_conditions(x, targets, user_input)

    def prompt_for_relational_conditions(self, x: Case, targets: List[ObjectPropertyTarget],
                                         user_input: Optional[str] = None) -> Dict[str, Condition]:
        """
        Prompt the user for relational conditions.

        :param x: The case to classify.
        :param targets: The target categories to compare the case with.
        :param user_input: The user input to parse. If None, the user is prompted for input.
        :return: The differentiating features as new rule conditions.
        """
        conditions = {}
        for target in targets:
            session = get_prompt_session_for_obj(x._obj)
            prompt_str = f"Give Conditions for {x._id}.{target.name}"
            user_input, tree = prompt_user_for_input(prompt_str, session, user_input=user_input)
            condition = Human.parse_relational_conditions(x, user_input, tree, target)
            conditions[target.name] = condition
            if not self.use_loaded_answers:
                self.all_expert_answers.append(user_input)
        return conditions

    @staticmethod
    def parse_relational_conditions(obj: Any, user_input: str, tree: ast.AST,
                                    target: ObjectPropertyTarget) -> Callable[[Case], bool]:
        """
        Parse the conditions from the user input.

        :param obj: The Object to classify.
        :param user_input: The input to parse.
        :param tree: The AST tree of the input.
        :param target: The target to classify.
        :return: The parsed conditions as a dictionary.
        """
        visitor = VariableVisitor()
        visitor.visit(tree)

        code = compile(tree, filename="<string>", mode="eval")

        class GeneratedCondition(Condition):

            def __init__(self):
                super().__init__(target.name, target.value, self.__call__)

            def __call__(self, case: Case, **kwargs) -> bool:
                try:
                    context = {}
                    for key in visitor.variables:
                        if key == "case" or (isinstance(case, Case) and key == case._id):
                            context[key] = case
                        elif key in dir(case):
                            context[key] = get_attribute_values(case, key)
                        elif get_property_name(obj, case) == key:
                            context[key] = case
                        else:
                            raise ValueError(f"Attribute {key} not found in the case {case}")
                    return eval(code, {"__builtins__": {"len": len}}, context)
                except Exception as e:
                    raise ValueError(f"Error during evaluation: {e}")

            def __str__(self):
                return user_input

        return GeneratedCondition()

    def ask_for_conditions(self, x: Case,
                           targets: Union[Attribute, List[Attribute]],
                           last_evaluated_rule: Optional[Rule] = None) \
            -> Dict[str, Condition]:
        targets = targets if isinstance(targets, list) else [targets]
        if last_evaluated_rule and not self.use_loaded_answers:
            action = "Refinement" if last_evaluated_rule.fired else "Alternative"
            print(f"{action} needed for rule:\n")

        all_attributes = self.get_all_attributes(x, last_evaluated_rule)

        all_names, max_len = x.get_all_names_and_max_len(all_attributes)

        if not self.use_loaded_answers:
            max_len = x.print_all_names(all_names, max_len, target_types=list(map(type, targets)))

            if last_evaluated_rule and last_evaluated_rule.fired:
                last_evaluated_rule.corner_case.print_values(all_names, is_corner_case=True,
                                                             ljust_sz=max_len)
            x.print_values(all_names, targets=targets, ljust_sz=max_len)

        if self.mode == RDRMode.Relational:
            return self._get_relational_conditions(x, targets, conditions_for="differentiating features")
        else:
            return self._get_conditions(all_names, conditions_for="differentiating features")

    def get_all_attributes(self, x: Case, last_evaluated_rule: Optional[Rule] = None) -> List[Attribute]:
        """
        Get all attributes for the case.

        :param x: The case to get the attributes for.
        :param last_evaluated_rule: The last evaluated rule.
        """
        if last_evaluated_rule and last_evaluated_rule.fired:
            all_attributes = last_evaluated_rule.corner_case._attributes_list + x._attributes_list
        else:
            if not self.use_loaded_answers:
                print("Please provide a rule for case:")
            all_attributes = x._attributes_list
        return all_attributes

    def ask_for_extra_conclusions(self, x: Case, current_conclusions: List[Attribute]) \
            -> Dict[Attribute, Dict[str, Condition]]:
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

    def ask_for_relational_conclusion(self, x: Case, for_attribute: Any) \
            -> Optional[Callable[[Case], None]]:
        """
        Ask the expert to provide a relational conclusion for the case.

        :param x: The case to classify.
        :param for_attribute: The attribute to provide the conclusion for.
        """
        all_names = self.get_and_print_all_names_and_conclusions(x)

        for_attribute_name = get_property_name(x._obj, for_attribute)

        if not hasattr(x, for_attribute_name):
            raise ValueError(f"Attribute {for_attribute_name} not found in the case")

        if not self.use_loaded_answers:
            prompt_str = f"Give Conclusion on {x.__class__.__name__}.{for_attribute_name}"
            session = get_prompt_session_for_obj(x)
            user_input, tree = prompt_user_for_input(prompt_str, session)
            self.all_expert_answers.append(user_input)
        else:
            user_input = self.all_expert_answers.pop(0)

        def apply_conclusion(case: Case) -> None:
            attr_value = parse_relational_conclusion(case, user_input)
            case[for_attribute_name] = attr_value
            return attr_value

        print(f"Evaluated expression: {apply_conclusion(x)}")

        return apply_conclusion

    def ask_for_conclusion(self, x: Case, current_conclusions: Optional[List[Attribute]] = None) -> Optional[Attribute]:
        """
        Ask the expert to provide a conclusion for the case.

        :param x: The case to classify.
        :param current_conclusions: The current conclusions for the case if any.
        """
        all_names = self.get_and_print_all_names_and_conclusions(x, current_conclusions)
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

    def get_and_print_all_names_and_conclusions(self, x: Case, current_conclusions: Optional[List[Attribute]] = None) \
            -> List[str]:
        """
        Get and print all names and conclusions for the case.

        :param x: The case to get the names and conclusions for.
        :param current_conclusions: The current conclusions for the case.
        :return: The list of all names.
        """
        conclusion_types = list(map(type, current_conclusions)) if current_conclusions else None
        all_names, max_len = x.get_all_names_and_max_len()
        if not self.use_loaded_answers:
            max_len = x.print_all_names(all_names, max_len, conclusion_types=conclusion_types)
            x.print_values(all_names, conclusions=current_conclusions, ljust_sz=max_len)
        return all_names

    def parse_conclusion(self, value: str) -> Attribute:
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
                raise ValueError(f"Attribute name {cat_name} should only contain alphabets")
            cat_value = True
        else:
            cat_name_value = value.split(":")
            cat_name = cat_name_value[0].strip(' "')
            if len(cat_name_value) == 2:
                cat_value = self.parse_value(cat_name_value[1])
            else:
                raise ValueError(f"Input format \"{value}\" is not correct")
        category = self.create_category_instance(cat_name, cat_value)
        return category

    def create_category_instance(self, cat_name: str, cat_value: Union[str, int, float, set]) -> Attribute:
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

    def get_category_type(self, cat_name: str) -> Optional[Type[Attribute]]:
        """
        Get the category type from the known categories.

        :param cat_name: The name of the category.
        :return: The category type.
        """
        cat_name = cat_name.lower()
        self.known_categories = get_all_subclasses(Attribute) if not self.known_categories else self.known_categories
        self.known_categories.update(Attribute._registry)
        category_type = None
        if cat_name in self.known_categories:
            category_type = self.known_categories[cat_name]
        return category_type

    def create_new_category_type(self, cat_name: str) -> Type[Attribute]:
        """
        Create a new category type.

        :param cat_name: The name of the category.
        :return: A new category type.
        """
        category_type: Type[Attribute] = type(cat_name, (Categorical,), {})
        if self.ask_if_category_is_mutually_exclusive(category_type.__name__):
            category_type._mutually_exclusive = True
        Attribute.register(category_type)
        return category_type

    def ask_if_category_is_mutually_exclusive(self, category_name: str) -> bool:
        """
        Ask the expert if the new category can have multiple values.

        :param category_name: The name of the category to ask about.
        """
        question = f"Can a case have multiple values of the new category {category_name}? (y/n):"
        return not self.ask_yes_no_question(question)

    def ask_if_conclusion_is_correct(self, x: Case, conclusion: Attribute,
                                     targets: Optional[List[Attribute]] = None,
                                     current_conclusions: Optional[List[Attribute]] = None) -> bool:
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
            x._conclusions = current_conclusions
            x._targets = targets
            question = f"Is the conclusion {conclusion} correct for the case (y/n):" \
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

    def parse_value(self, value: str) -> Union[str, int, float, set]:
        """
        Parse the value from the user input.

        :param value: The value to parse.
        :return: The parsed value as a string, float or set of string or float.
        """
        if self.is_set(value):
            set_values = value[1:-1].split(",")
            for i, val in enumerate(set_values):
                set_values[i] = self.parse_string_int_or_float(val)
            parsed_value = set(set_values)
        else:
            parsed_value = self.parse_string_int_or_float(value)
        return parsed_value

    def parse_string_int_or_float(self, val: str) -> Union[str, int, float]:
        """
        Parse the value to a string or a float.

        :param val: The value to parse.
        :return: The parsed value as a string or a float.
        """
        if self.is_string(val):
            return val.strip(' "' + "'")
        elif val.isdigit():
            return int(val)
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
            -> Tuple[List[str], Optional[str], Optional[str], Optional[Operator]]:
        """
        Validate the input and get error messages.

        :param all_names: list of all attribute names.
        :param rule: The rule to validate.
        :return: list of error messages, and the name, value and operator of the rule.
        """
        try:
            operator = Operator.parse_operators(rule)
            operator = operator[0]
            name = operator.arg_names[0]
            value = operator.arg_names[1]
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
