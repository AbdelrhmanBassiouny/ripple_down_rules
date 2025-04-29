import ast
import logging
from _ast import AST

from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config import Config
from typing_extensions import List, Optional, Tuple, Dict

from .datastructures.enums import PromptFor
from .datastructures.callable_expression import CallableExpression, parse_string_to_expression
from .datastructures.dataclasses import CaseQuery
from .utils import extract_dependencies, contains_return_statement, make_set


class CustomInteractiveShell(InteractiveShellEmbed):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.all_lines = []

    def run_cell(self, raw_cell: str, **kwargs):
        """
        Override the run_cell method to capture return statements.
        """
        if contains_return_statement(raw_cell):
            self.all_lines.append(raw_cell)
            print("Exiting shell on `return` statement.")
            self.history_manager.store_inputs(line_num=self.execution_count, source=raw_cell)
            self.ask_exit()
            return None
        result = super().run_cell(raw_cell, **kwargs)
        if result.error_in_exec is None and result.error_before_exec is None:
            self.all_lines.append(raw_cell)
        return result


class IPythonShell:
    """
    Create an embedded Ipython shell that can be used to prompt the user for input.
    """

    def __init__(self, scope: Optional[Dict] = None, header: Optional[str] = None):
        """
        Initialize the Ipython shell with the given scope and header.

        :param scope: The scope to use for the shell.
        :param header: The header to display when the shell is started.
        """
        self.scope: Dict = scope or {}
        self.header: str = header or ">>> Embedded Ipython Shell"
        self.user_input: Optional[str] = None
        self.shell: CustomInteractiveShell = self._init_shell()
        self.all_code_lines: List[str] = []

    def _init_shell(self):
        """
        Initialize the Ipython shell with a custom configuration.
        """
        cfg = Config()
        shell = CustomInteractiveShell(config=cfg, user_ns=self.scope, banner1=self.header)
        return shell

    def run(self):
        """
        Run the embedded shell.
        """
        while True:
            try:
                self.shell()
                self.update_user_input_from_code_lines()
                break
            except Exception as e:
                logging.error(e)
                print(e)

    def update_user_input_from_code_lines(self):
        """
        Update the user input from the code lines captured in the shell.
        """
        if len(self.shell.all_lines) == 1 and self.shell.all_lines[0].replace('return', '').strip() == '':
            self.user_input = None
        else:
            self.all_code_lines = extract_dependencies(self.shell.all_lines)
            if len(self.all_code_lines) == 1:
                if self.all_code_lines[0].strip() == '':
                    self.user_input = None
                else:
                    self.user_input = self.all_code_lines[0].replace('return', '').strip()
            else:
                self.user_input = f"def _get_value(case):\n    "
                for cl in self.all_code_lines:
                    sub_code_lines = cl.split('\n')
                    self.user_input += '\n    '.join(sub_code_lines) + '\n    '


def prompt_user_for_expression(case_query: CaseQuery, prompt_for: PromptFor, prompt_str: Optional[str] = None)\
        -> Tuple[Optional[str], Optional[CallableExpression]]:
    """
    Prompt the user for an executable python expression to the given case query.

    :param case_query: The case query to prompt the user for.
    :param prompt_for: The type of information ask user about.
    :param prompt_str: The prompt string to display to the user.
    :return: A callable expression that takes a case and executes user expression on it.
    """
    while True:
        user_input, expression_tree = prompt_user_about_case(case_query, prompt_for, prompt_str)
        if user_input is None:
            if prompt_for == PromptFor.Conclusion:
                print("No conclusion provided. Exiting.")
                return None, None
            else:
                print("Conditions must be provided. Please try again.")
                continue
        conclusion_type = bool if prompt_for == PromptFor.Conditions else case_query.attribute_type
        callable_expression = CallableExpression(user_input, conclusion_type, expression_tree=expression_tree,
                                                 scope=case_query.scope)
        try:
            callable_expression(case_query.case)
            break
        except Exception as e:
            logging.error(e)
            print(e)
    return user_input, callable_expression


def prompt_user_about_case(case_query: CaseQuery, prompt_for: PromptFor,
                           prompt_str: Optional[str] = None) -> Tuple[Optional[str], Optional[AST]]:
    """
    Prompt the user for input.

    :param case_query: The case query to prompt the user for.
    :param prompt_for: The type of information the user should provide for the given case.
    :param prompt_str: The prompt string to display to the user.
    :return: The user input, and the executable expression that was parsed from the user input.
    """
    if prompt_str is None:
        prompt_str = f"Give {prompt_for} for {case_query.name}"
    scope = {'case': case_query.case, **case_query.scope}
    shell = IPythonShell(scope=scope, header=prompt_str)
    return prompt_user_input_and_parse_to_expression(shell=shell)


def prompt_user_input_and_parse_to_expression(shell: Optional[IPythonShell] = None,
                                              user_input: Optional[str] = None)\
        -> Tuple[Optional[str], Optional[ast.AST]]:
    """
    Prompt the user for input.

    :param shell: The Ipython shell to use for prompting the user.
    :param user_input: The user input to use. If given, the user input will be used instead of prompting the user.
    :return: The user input and the AST tree.
    """
    while True:
        if user_input is None:
            shell = IPythonShell() if shell is None else shell
            shell.run()
            user_input = shell.user_input
            if user_input is None:
                return None, None
            print(user_input)
        try:
            return user_input, parse_string_to_expression(user_input)
        except Exception as e:
            msg = f"Error parsing expression: {e}"
            logging.error(msg)
            print(msg)
            user_input = None
