from __future__ import annotations

import ast
import json
import re
from abc import ABC, abstractmethod

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from typing_extensions import Optional, Dict, TYPE_CHECKING, List, Tuple, Type, Union, Any, Sequence, Callable

from .datastructures import str_to_operator_fn, Condition, Case, Attribute, Operator, RDRMode
from .failures import InvalidOperator
from .utils import get_all_subclasses, get_attribute_values, get_completions, get_property_name

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

    def ask_for_relational_conclusion(self, x: Case, for_attribute: Union[str, Attribute, Sequence[Attribute]])\
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
        # if not self.use_loaded_answers:
        #     print(f"Please provide comma separated conditions for {conditions_for}:")
        while True:
            if self.use_loaded_answers:
                value = self.all_expert_answers.pop(0)
            else:
                # Initialze prompt session and get conditions from user written input
                session = self.get_prompt_session_for_obj(x.obj)
                rule_conditions: Dict[str, Condition] = None
                while True:
                    user_input = session.prompt(
                        f"\nGive Conditions for {x.__class__.__name__}.{targets[0].name} >>> ")
                    if user_input.lower() in ['exit', 'quit', '']:
                        break
                    print(f"Evaluating: {user_input}")
                    try:
                        # Parse the input into an AST
                        tree = ast.parse(user_input, mode='eval')
                        print(f"AST parsed successfully: {ast.dump(tree)}")

                        # Step 2: Extract variable names from the tree
                        class VariableVisitor(ast.NodeVisitor):
                            def __init__(self):
                                self.variables = set()
                                self.compares = list()
                                self.all = list()

                            def visit_BinOp(self, node):
                                self.all.append(node)
                                self.generic_visit(node)

                            def visit_BoolOp(self, node):
                                self.all.append(node)
                                self.generic_visit(node)

                            def visit_Compare(self, node):
                                self.all.append(node)
                                self.compares.append([node.left, node.ops[0], node.comparators[0]])
                                self.generic_visit(node)

                            def visit_Name(self, node):
                                if node.id not in dir(__builtins__):
                                    self.variables.add(node.id)
                                self.generic_visit(node)

                        visitor = VariableVisitor()
                        visitor.visit(tree)

                        # Step 3: Compile the AST into a function
                        code = compile(tree, filename="<string>", mode="eval")
                        arg_names = ", ".join(visitor.variables)

                        # Step 4: Create a callable function
                        def generated_function(**kwargs):
                            try:
                                # Pass only the required arguments to eval
                                context = {key: kwargs[key] for key in visitor.variables if key in kwargs}
                                return eval(code, {"__builtins__": None}, context)
                            except Exception as e:
                                raise ValueError(f"Error during evaluation: {e}")

                        def rule_condition(x: Case) -> None:
                            return self.parse_relational_conditions(x.obj, user_input)

                        print(f"Evaluated expression: {rule_condition(x)}")
                    except SyntaxError as e:
                        print(f"Syntax error: {e}")
                return rule_conditions

    def parse_relational_conditions(self, x: Case, user_input: str) -> Dict[str, Condition]:
        """
        Parse the conditions from the user input.

        :param x: The case to classify.
        :param user_input: The input to parse.
        :return: The parsed conditions as a dictionary.
        """
        rule_conditions = {}
        operators = get_all_subclasses(Operator)
        rules = re.split()
        all_messages = []
        for rule in rules:
            rule_conditions, messages = self.parse_rule(rule, all_names)
            all_messages += messages if messages else []
            rule_conditions.update(rule_conditions)
        if all_messages:
            print("\n".join(all_messages))
        return rule_conditions

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
            all_attributes = last_evaluated_rule.corner_case.attributes_list + x.attributes_list
        else:
            if not self.use_loaded_answers:
                print("Please provide a rule for case:")
            all_attributes = x.attributes_list
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

    def ask_for_relational_conclusion(self, x: Case, for_attribute: Any)\
            -> Optional[Callable[[Case], None]]:
        """
        Ask the expert to provide a relational conclusion for the case.

        :param x: The case to classify.
        :param for_attribute: The attribute to provide the conclusion for.
        """
        all_names = self.get_and_print_all_names_and_conclusions(x)
        session = self.get_prompt_session_for_obj(x.obj)

        for_attribute_name = get_property_name(x.obj, for_attribute)

        if not hasattr(x.obj, for_attribute_name):
            raise ValueError(f"Attribute {for_attribute_name} not found in the case")
        apply_conclusion = None
        while True:
            user_input = session.prompt(f"\nGive Conclusion on {x.__class__.__name__}.{for_attribute_name} >>> ")
            if user_input.lower() in ['exit', 'quit', '']:
                break
            print(f"Evaluating: {user_input}")
            try:
                # Parse the input into an AST
                tree = ast.parse(user_input, mode='eval')
                print(f"AST parsed successfully: {ast.dump(tree)}")

                def apply_conclusion(x: Case) -> None:
                    attr_value = self.parse_relational_conclusion(x.obj, user_input)
                    x[for_attribute_name] = attr_value
                print(f"Evaluated expression: {apply_conclusion(x)}")
            except SyntaxError as e:
                print(f"Syntax error: {e}")
        return apply_conclusion

    @staticmethod
    def get_prompt_session_for_obj(obj: Any) -> PromptSession:
        """
        Get a prompt session for an object.

        :param obj: The object to get the prompt session for.
        :return: The prompt session.
        """
        completions = get_completions(obj)
        completer = WordCompleter(completions)
        session = PromptSession(completer=completer)
        return session

    @staticmethod
    def get_attribute_name(attribute: Union[str, Attribute, Sequence[Attribute]]) -> str:
        """
        Get the attribute name.

        :param attribute: The attribute object to get the attribute name from.
        :return: The attribute name.
        """
        if isinstance(attribute, type):
            attribute_name = attribute.__name__
        elif hasattr(attribute, "__iter__") and not isinstance(attribute, str):
            if isinstance(attribute, set):
                attribute = list(attribute)
            attribute_name = attribute[0].__class__.__name__
        elif isinstance(attribute, Attribute):
            attribute_name = attribute.__class__.__name__
        elif isinstance(attribute, str):
            attribute_name = attribute
        else:
            raise ValueError(f"Attribute {attribute} is not valid, expected an str, an Attribute, an Attribute Type"
                             f" or a list of same type Attributes")
        return attribute_name

    @staticmethod
    def parse_relational_conclusion(x: Case, conclusion: str) -> Any:
        """
        Parse a relational conclusion from a string and get the attribute values equivalent to the conclusion from the case.

        :param x: The case to get the attribute values from.
        :param conclusion: The conclusion to parse.
        """
        attr_chain = conclusion.split('.')
        user_attr = attr_chain[0]
        user_sub_attr = attr_chain[1] if len(attr_chain) > 1 else None
        # Evaluate expression
        attr = getattr(x, user_attr)
        if user_sub_attr:
            attr = get_attribute_values(attr, user_sub_attr)
        attr = set().union(*attr) if hasattr(attr, "__iter__") and not isinstance(attr, str) else attr
        return attr

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

    def get_and_print_all_names_and_conclusions(self, x: Case, current_conclusions: Optional[List[Attribute]] = None)\
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
            category_type = self.create_new_category_type(cat_name, cat_value)
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

    def create_new_category_type(self, cat_name: str, cat_value: Union[str, int, float, set]) -> Type[Attribute]:
        """
        Create a new category type.

        :param cat_name: The name of the category.
        :param cat_value: The value of the category.
        :return: A new category type.
        """
        category_type: Type[Attribute] = type(cat_name, (Attribute,), {})
        if self.ask_if_category_is_mutually_exclusive(category_type.__name__):
            category_type.mutually_exclusive = True
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
            x.conclusions = current_conclusions
            x.targets = targets
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
