from __future__ import annotations

import json
from abc import ABC, abstractmethod

from typing_extensions import Optional, TYPE_CHECKING, List

from .datastructures.callable_expression import CallableExpression
from .datastructures.enums import PromptFor
from .datastructures.dataclasses import CaseQuery
from .datastructures.case import show_current_and_corner_cases
from .user_interface.gui import RDRCaseViewer
from .user_interface.prompt import UserPrompt

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

    def __init__(self, use_loaded_answers: bool = False, append: bool = False):
        self.all_expert_answers = []
        self.use_loaded_answers = use_loaded_answers
        self.append = append

    @abstractmethod
    def ask_for_conditions(self, case_query: CaseQuery, last_evaluated_rule: Optional[Rule] = None) \
            -> CallableExpression:
        """
        Ask the expert to provide the differentiating features between two cases or unique features for a case
        that doesn't have a corner case to compare to.

        :param case_query: The case query containing the case to classify and the required target.
        :param last_evaluated_rule: The last evaluated rule.
        :return: The differentiating features as new rule conditions.
        """
        pass

    @abstractmethod
    def ask_for_conclusion(self, case_query: CaseQuery) -> Optional[CallableExpression]:
        """
        Ask the expert to provide a relational conclusion for the case.

        :param case_query: The case query containing the case to find a conclusion for.
        :return: A callable expression that can be called with a new case as an argument.
        """


class Human(Expert):
    """
    The Human Expert class, an expert that asks the human to provide differentiating features and conclusions.
    """

    def __init__(self, use_loaded_answers: bool = False, append: bool = False, viewer: Optional[RDRCaseViewer] = None):
        """
        Initialize the Human expert.

        :param viewer: The RDRCaseViewer instance to use for prompting the user.
        """
        super().__init__(use_loaded_answers=use_loaded_answers, append=append)
        self.user_prompt = UserPrompt(viewer)

    def save_answers(self, path: str):
        """
        Save the expert answers to a file.

        :param path: The path to save the answers to.
        """
        if self.append:
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

    def ask_for_conditions(self, case_query: CaseQuery,
                           last_evaluated_rule: Optional[Rule] = None) \
            -> CallableExpression:
        if not self.use_loaded_answers and self.user_prompt.viewer is None:
            show_current_and_corner_cases(case_query.case, {case_query.attribute_name: case_query.target_value},
                                          last_evaluated_rule=last_evaluated_rule)
        return self._get_conditions(case_query)

    def _get_conditions(self, case_query: CaseQuery) \
            -> CallableExpression:
        """
        Ask the expert to provide the differentiating features between two cases or unique features for a case
        that doesn't have a corner case to compare to.

        :param case_query: The case query containing the case to classify.
        :return: The differentiating features as new rule conditions.
        """
        user_input = None
        if self.use_loaded_answers and len(self.all_expert_answers) == 0 and self.append:
            self.use_loaded_answers = False
        if self.use_loaded_answers:
            user_input = self.all_expert_answers.pop(0)
        if user_input:
            condition = CallableExpression(user_input, bool, scope=case_query.scope)
        else:
            user_input, condition = self.user_prompt.prompt_user_for_expression(case_query, PromptFor.Conditions)
        if not self.use_loaded_answers:
            self.all_expert_answers.append(user_input)
        case_query.conditions = condition
        return condition

    def ask_for_conclusion(self, case_query: CaseQuery) -> Optional[CallableExpression]:
        """
        Ask the expert to provide a conclusion for the case.

        :param case_query: The case query containing the case to find a conclusion for.
        :return: The conclusion for the case as a callable expression.
        """
        expression: Optional[CallableExpression] = None
        if self.use_loaded_answers and len(self.all_expert_answers) == 0 and self.append:
            self.use_loaded_answers = False
        if self.use_loaded_answers:
            expert_input = self.all_expert_answers.pop(0)
            if expert_input is not None:
                expression = CallableExpression(expert_input, case_query.attribute_type,
                                                scope=case_query.scope,
                                                 mutually_exclusive=case_query.mutually_exclusive)
        else:
            if self.user_prompt.viewer is None:
                show_current_and_corner_cases(case_query.case)
            expert_input, expression = self.user_prompt.prompt_user_for_expression(case_query, PromptFor.Conclusion)
            self.all_expert_answers.append(expert_input)
        case_query.target = expression
        return expression
