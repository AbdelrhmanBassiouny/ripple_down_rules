from __future__ import annotations

import json
from abc import ABC, abstractmethod

from sqlalchemy.orm import DeclarativeBase as SQLTable, MappedColumn as SQLColumn, Session
from typing_extensions import Optional, Dict, TYPE_CHECKING, List, Tuple, Type, Union, Any

from .datastructures import (Case, PromptFor, CallableExpression, Column, CaseQuery)
from .datastructures.table import show_current_and_corner_cases
from .prompt import prompt_user_for_expression, prompt_user_about_case
from .utils import get_all_subclasses

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
    known_categories: Optional[Dict[str, Type[Column]]] = None
    """
    The known categories (i.e. Column types) to use.
    """

    @abstractmethod
    def ask_for_conditions(self, x: Case, targets: List[Column], last_evaluated_rule: Optional[Rule] = None) \
            -> CallableExpression:
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
    def ask_for_extra_conclusions(self, x: Case, current_conclusions: List[Column]) \
            -> Dict[Column, CallableExpression]:
        """
        Ask the expert to provide extra conclusions for a case by providing a pair of category and conditions for
        that category.

        :param x: The case to classify.
        :param current_conclusions: The current conclusions for the case.
        :return: The extra conclusions for the case.
        """
        pass

    @abstractmethod
    def ask_if_conclusion_is_correct(self, x: Case, conclusion: Column,
                                     targets: Optional[List[Column]] = None,
                                     current_conclusions: Optional[List[Column]] = None) -> bool:
        """
        Ask the expert if the conclusion is correct.

        :param x: The case to classify.
        :param conclusion: The conclusion to check.
        :param targets: The target categories to compare the case with.
        :param current_conclusions: The current conclusions for the case.
        """
        pass

    def ask_for_conclusion(self, case_query: CaseQuery,
                           session: Optional[Session] = None) -> Optional[CallableExpression]:
        """
        Ask the expert to provide a relational conclusion for the case.

        :param case_query: The case query containing the case to find a conclusion for.
        :param session: The sqlalchemy orm session to use if the case is a Table.
        :return: A callable expression that can be called with a new case as an argument.
        """


class Human(Expert):
    """
    The Human Expert class, an expert that asks the human to provide differentiating features and conclusions.
    """

    def __init__(self, use_loaded_answers: bool = False, session: Optional[Session] = None):
        self.all_expert_answers = []
        self.use_loaded_answers = use_loaded_answers
        self.session = session

    def save_answers(self, path: str, append: bool = False):
        """
        Save the expert answers to a file.

        :param path: The path to save the answers to.
        :param append: A flag to indicate if the answers should be appended to the file or not.
        """
        if append:
            # read the file and append the new answers
            with open(path + '.json', "r") as f:
                all_answers = json.load(f)
                all_answers.extend(self.all_expert_answers)
            with open(path + '.json', "w") as f:
                json.dump(all_answers, f)
        else:
            with open(path + '.json', "w") as f:
                json.dump(self.all_expert_answers, f)

    def load_answers(self, path: str):
        """
        Load the expert answers from a file.

        :param path: The path to load the answers from.
        """
        with open(path + '.json', "r") as f:
            self.all_expert_answers = json.load(f)

    def ask_for_conditions(self, case: Case,
                           targets: Union[List[Column], List[Column]],
                           last_evaluated_rule: Optional[Rule] = None) \
            -> CallableExpression:
        if not self.use_loaded_answers:
            show_current_and_corner_cases(case, targets, last_evaluated_rule=last_evaluated_rule)
        return self._get_conditions(case, targets)

    def _get_conditions(self, case: Case, targets: List[Column]) \
            -> CallableExpression:
        """
        Ask the expert to provide the differentiating features between two cases or unique features for a case
        that doesn't have a corner case to compare to.

        :param case: The case to classify.
        :param targets: The target categories to compare the case with.
        :return: The differentiating features as new rule conditions.
        """
        targets = targets if isinstance(targets, list) else [targets]
        condition = None
        for target in targets:
            target_name = target.__class__.__name__
            user_input = None
            if self.use_loaded_answers:
                user_input = self.all_expert_answers.pop(0)
            if user_input:
                condition = CallableExpression(user_input, bool, session=self.session)
            else:
                user_input, condition = prompt_user_for_expression(case, PromptFor.Conditions, target_name, bool)
            if not self.use_loaded_answers:
                self.all_expert_answers.append(user_input)
        return condition

    def ask_for_extra_conclusions(self, case: Case, current_conclusions: List[Column]) \
            -> Dict[Column, CallableExpression]:
        """
        Ask the expert to provide extra conclusions for a case by providing a pair of category and conditions for
        that category.

        :param case: The case to classify.
        :param current_conclusions: The current conclusions for the case.
        :return: The extra conclusions for the case.
        """
        extra_conclusions = {}
        while True:
            category = self.ask_for_conclusion(CaseQuery(case), current_conclusions)
            if not category:
                break
            extra_conclusions[category] = self._get_conditions(case, category)
        return extra_conclusions

    def ask_for_conclusion(self, case_query: CaseQuery,
                           current_conclusions: Optional[List[Any]] = None)\
            -> Optional[CallableExpression]:
        """
        Ask the expert to provide a conclusion for the case.

        :param case_query: The case query containing the case to find a conclusion for.
        :param current_conclusions: The current conclusions for the case if any.
        :return: The conclusion for the case.
        """
        case = case_query.case
        attribute_name = case_query.attribute_name
        attribute_type = case_query.attribute_type
        if self.use_loaded_answers:
            expert_input = self.all_expert_answers.pop(0)
            expression = CallableExpression(expert_input, conclusion_type=attribute_type, session=self.session)
        else:
            show_current_and_corner_cases(case, current_conclusions=current_conclusions)
            expert_input, expression = prompt_user_for_expression(case, PromptFor.Conclusion, attribute_name,
                                                                  attribute_type)
            self.all_expert_answers.append(expert_input)
        return expression

    def create_category_instance(self, cat_name: str, cat_value: Union[str, int, float, set]) -> Column:
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

    def get_category_type(self, cat_name: str) -> Optional[Type[Column]]:
        """
        Get the category type from the known categories.

        :param cat_name: The name of the category.
        :return: The category type.
        """
        cat_name = cat_name.lower()
        self.known_categories = get_all_subclasses(Column) if not self.known_categories else self.known_categories
        self.known_categories.update(Column.registry)
        category_type = None
        if cat_name in self.known_categories:
            category_type = self.known_categories[cat_name]
        return category_type

    def create_new_category_type(self, cat_name: str) -> Type[Column]:
        """
        Create a new category type.

        :param cat_name: The name of the category.
        :return: A new category type.
        """
        if self.ask_if_category_is_mutually_exclusive(cat_name):
            category_type: Type[Column] = Column.create(cat_name, set(), mutually_exclusive=True)
        else:
            category_type: Type[Column] = Column.create(cat_name, set())
        return category_type

    def ask_if_category_is_mutually_exclusive(self, category_name: str) -> bool:
        """
        Ask the expert if the new category can have multiple values.

        :param category_name: The name of the category to ask about.
        """
        question = f"Can a case have multiple values of the new category {category_name}? (y/n):"
        return not self.ask_yes_no_question(question)

    def ask_if_conclusion_is_correct(self, x: Case, conclusion: Column,
                                     targets: Optional[List[Column]] = None,
                                     current_conclusions: Optional[List[Column]] = None) -> bool:
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
