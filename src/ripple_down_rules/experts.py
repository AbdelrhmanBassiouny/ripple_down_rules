from __future__ import annotations

import ast
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod

from typing_extensions import Optional, TYPE_CHECKING, List

from .datastructures.callable_expression import CallableExpression
from .datastructures.enums import PromptFor
from .datastructures.dataclasses import CaseQuery
from .datastructures.case import show_current_and_corner_cases
from .utils import extract_imports, extract_function_source, get_imports_from_scope, encapsulate_user_input

try:
    from .user_interface.gui import RDRCaseViewer
except ImportError as e:
    RDRCaseViewer = None
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

    def __init__(self, use_loaded_answers: bool = True,
                 append: bool = False,
                 answers_save_path: Optional[str] = None):
        self.all_expert_answers = []
        self.use_loaded_answers = use_loaded_answers
        self.append = append
        self.answers_save_path = answers_save_path
        if answers_save_path is not None:
            if use_loaded_answers:
                self.load_answers(answers_save_path)
            else:
                os.remove(answers_save_path + '.py')
            self.append = True

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

    def clear_answers(self, path: Optional[str] = None):
        """
        Clear the expert answers.

        :param path: The path to clear the answers from. If None, the answers will be cleared from the
                     answers_save_path attribute.
        """
        if path is None and self.answers_save_path is None:
            raise ValueError("No path provided to clear expert answers, either provide a path or set the "
                                "answers_save_path attribute.")
        if path is None:
            path = self.answers_save_path
        if os.path.exists(path + '.json'):
            os.remove(path + '.json')
        if os.path.exists(path + '.py'):
            os.remove(path + '.py')
        self.all_expert_answers = []

    def save_answers(self, path: Optional[str] = None):
        """
        Save the expert answers to a file.

        :param path: The path to save the answers to.
        """
        if path is None and self.answers_save_path is None:
            raise ValueError("No path provided to save expert answers, either provide a path or set the "
                                "answers_save_path attribute.")
        if path is None:
            path = self.answers_save_path
        is_json = os.path.exists(path + '.json')
        if is_json:
            self._save_to_json(path)
        else:
            self._save_to_python(path)

    def _save_to_json(self, path: str):
        """
        Save the expert answers to a JSON file.

        :param path: The path to save the answers to.
        """
        all_answers = self.all_expert_answers
        if self.append and os.path.exists(path + '.json'):
            # read the file and append the new answers
            with open(path + '.json', "r") as f:
                old_answers = json.load(f)
                all_answers = old_answers + all_answers
        with open(path + '.json', "w") as f:
            json.dump(all_answers, f)

    def _save_to_python(self, path: str):
        """
        Save the expert answers to a Python file.

        :param path: The path to save the answers to.
        """
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name + '/__init__.py'):
            os.makedirs(dir_name, exist_ok=True)
            with open(dir_name + '/__init__.py', 'w') as f:
                f.write('# This is an empty init file to make the directory a package.\n')
        action = 'w' if not self.append else 'a'
        with open(path + '.py', action) as f:
            for scope, func_source in self.all_expert_answers:
                if len(scope) > 0:
                    imports = '\n'.join(get_imports_from_scope(scope)) + '\n\n\n'
                else:
                    imports = ''
                if func_source is not None:
                    uid = uuid.uuid4().hex
                    func_source = encapsulate_user_input(func_source, CallableExpression.encapsulating_function + f'_{uid}')
                else:
                    func_source = 'pass  # No user input provided for this case.\n'
                f.write(imports + func_source + '\n' + '\n\n\n\'===New Answer===\'\n\n\n')

    def load_answers(self, path: Optional[str] = None):
        """
        Load the expert answers from a file.

        :param path: The path to load the answers from.
        """
        if path is None and self.answers_save_path is None:
            raise ValueError("No path provided to load expert answers from, either provide a path or set the "
                                "answers_save_path attribute.")
        if path is None:
            path = self.answers_save_path
        is_json = os.path.exists(path + '.json')
        if is_json:
            self._load_answers_from_json(path)
        elif os.path.exists(path + '.py'):
            self._load_answers_from_python(path)

    def _load_answers_from_json(self, path: str):
        """
        Load the expert answers from a JSON file.

        :param path: The path to load the answers from.
        """
        with open(path + '.json', "r") as f:
            all_answers = json.load(f)
        self.all_expert_answers = [({}, answer) for answer in all_answers]

    def _load_answers_from_python(self, path: str):
        """
        Load the expert answers from a Python file.

        :param path: The path to load the answers from.
        """
        file_path = path + '.py'
        with open(file_path, "r") as f:
            all_answers = f.read().split('\n\n\n\'===New Answer===\'\n\n\n')[:-1]
        all_function_sources = list(extract_function_source(file_path, []).values())
        all_function_sources_names = list(extract_function_source(file_path, []).keys())
        for i, answer in enumerate(all_answers):
            answer = answer.strip('\n').strip()
            if 'def ' not in answer and 'pass' in answer:
                self.all_expert_answers.append(({}, None))
                continue
            scope = extract_imports(tree=ast.parse(answer))
            function_source = all_function_sources[i].replace(all_function_sources_names[i],
                                                              CallableExpression.encapsulating_function_name)
            self.all_expert_answers.append((scope, function_source))


class Human(Expert):
    """
    The Human Expert class, an expert that asks the human to provide differentiating features and conclusions.
    """

    def __init__(self, viewer: Optional[RDRCaseViewer] = None, **kwargs):
        """
        Initialize the Human expert.

        :param viewer: The RDRCaseViewer instance to use for prompting the user.
        """
        super().__init__(**kwargs)
        self.user_prompt = UserPrompt(viewer)

    def ask_for_conditions(self, case_query: CaseQuery,
                           last_evaluated_rule: Optional[Rule] = None) \
            -> CallableExpression:
        if (not self.use_loaded_answers or len(self.all_expert_answers) == 0) and self.user_prompt.viewer is None:
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
            try:
                loaded_scope, user_input = self.all_expert_answers.pop(0)
            except IndexError:
                self.use_loaded_answers = False
        if user_input is not None:
            condition = CallableExpression(user_input, bool, scope=case_query.scope)
        else:
            user_input, condition = self.user_prompt.prompt_user_for_expression(case_query, PromptFor.Conditions)
        if not self.use_loaded_answers:
            self.all_expert_answers.append((condition.scope, user_input))
            if self.answers_save_path is not None:
                self.save_answers()
        case_query.conditions = condition
        return condition

    def ask_for_conclusion(self, case_query: CaseQuery) -> Optional[CallableExpression]:
        """
        Ask the expert to provide a conclusion for the case.

        :param case_query: The case query containing the case to find a conclusion for.
        :return: The conclusion for the case as a callable expression.
        """
        expression: Optional[CallableExpression] = None
        expert_input: Optional[str] = None
        if self.use_loaded_answers and len(self.all_expert_answers) == 0 and self.append:
            self.use_loaded_answers = False
        if self.use_loaded_answers:
            try:
                loaded_scope, expert_input = self.all_expert_answers.pop(0)
                if expert_input is not None:
                    expression = CallableExpression(expert_input, case_query.attribute_type,
                                                    scope=case_query.scope,
                                                    mutually_exclusive=case_query.mutually_exclusive)
            except IndexError:
                self.use_loaded_answers = False
        if not self.use_loaded_answers:
            if self.user_prompt.viewer is None:
                show_current_and_corner_cases(case_query.case)
            expert_input, expression = self.user_prompt.prompt_user_for_expression(case_query, PromptFor.Conclusion)
            if expression is None:
                self.all_expert_answers.append(({}, None))
            else:
                self.all_expert_answers.append((expression.scope, expert_input))
            if self.answers_save_path is not None:
                self.save_answers()
        case_query.target = expression
        return expression
