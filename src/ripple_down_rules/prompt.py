import ast
import logging
from _ast import AST

from IPython.core.interactiveshell import ExecutionInfo
from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config import Config
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from sqlalchemy.orm import DeclarativeBase as SQLTable, Session
from typing_extensions import Any, List, Optional, Tuple, Dict, Union, Type

from .datastructures import Case, PromptFor, CallableExpression, create_case, parse_string_to_expression, CaseQuery
from .utils import capture_variable_assignment, extract_dependencies, contains_return_statement


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
            self.ask_exit()
            return None
        result = super().run_cell(raw_cell, **kwargs)
        if not result.error_in_exec:
            self.all_lines.append(raw_cell)
        return result


class IpythonShell:
    """
    Create an embedded Ipython shell that can be used to prompt the user for input.
    """
    def __init__(self, prompt_for: PromptFor, scope: Optional[Dict] = None, header: Optional[str] = None):
        """
        Initialize the Ipython shell with the given scope and header.

        :param scope: The scope to use for the shell.
        :param header: The header to display when the shell is started.
        """
        self.prompt_for: PromptFor = prompt_for
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
        self.shell()
        self.all_code_lines = extract_dependencies(self.shell.all_lines)
        self.user_input = f"def _get_value(case):\n    "
        self.user_input += '\n    '.join(self.all_code_lines)


def prompt_user_for_expression(case_query: CaseQuery, prompt_for: PromptFor,
                               session: Optional[Session] = None) -> Tuple[str, CallableExpression]:
    """
    Prompt the user for an executable python expression to the given case query.

    :param case_query: The case query to prompt the user for.
    :param prompt_for: The type of information ask user about.
    :param session: The sqlalchemy orm session.
    :return: A callable expression that takes a case and executes user expression on it.
    """
    while True:
        user_input, expression_tree = prompt_user_about_case(case_query, prompt_for)
        conclusion_type = bool if prompt_for == PromptFor.Conditions else case_query.attribute_type
        callable_expression = CallableExpression(user_input, conclusion_type, expression_tree=expression_tree,
                                                 scope=case_query.scope, session=session)
        try:
            callable_expression(case_query.case)
            break
        except Exception as e:
            logging.error(e)
            print(e)
    return user_input, callable_expression


def prompt_user_about_case(case_query: CaseQuery, prompt_for: PromptFor) -> Tuple[str, AST]:
    """
    Prompt the user for input.

    :param case_query: The case query to prompt the user for.
    :param prompt_for: The type of information the user should provide for the given case.
    :return: The user input, and the executable expression that was parsed from the user input.
    """
    prompt_str = f"Give {prompt_for} for {case_query.name}"
    scope = {'case': case_query.case, **case_query.scope}
    shell = IpythonShell(prompt_for, scope=scope, header=prompt_str)
    user_input, expression_tree = prompt_user_input_and_parse_to_expression(shell=shell)
    return user_input, expression_tree


def get_completions(obj: Any) -> List[str]:
    """
    Get all completions for the object. This is used in the python prompt shell to provide completions for the user.

    :param obj: The object to get completions for.
    :return: A list of completions.
    """
    # Define completer with all object attributes and comparison operators
    completions = ['==', '!=', '>', '<', '>=', '<=', 'in', 'not', 'and', 'or', 'is']
    completions += ["isinstance(", "issubclass(", "type(", "len(", "hasattr(", "getattr(", "setattr(", "delattr("]
    completions += list(create_case(obj).keys())
    return completions


def prompt_user_input_and_parse_to_expression(shell: Optional[IpythonShell] = None,
                                              user_input: Optional[str] = None) -> Tuple[str, ast.AST]:
    """
    Prompt the user for input.

    :param shell: The Ipython shell to use for prompting the user.
    :param user_input: The user input to use. If given, the user input will be used instead of prompting the user.
    :return: The user input and the AST tree.
    """
    while True:
        if user_input is None:
            shell = IpythonShell() if shell is None else shell
            shell.run()
            user_input = shell.user_input
        try:
            return user_input, parse_string_to_expression(user_input)
        except Exception as e:
            msg = f"Error parsing expression: {e}"
            logging.error(msg)
            print(msg)
            user_input = None


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
