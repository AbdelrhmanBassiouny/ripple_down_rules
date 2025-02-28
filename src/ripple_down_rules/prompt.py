import ast
import logging
from _ast import AST

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from sqlalchemy.orm import DeclarativeBase as Table, Session
from typing_extensions import Any, List, Optional, Tuple, Dict, Union, Type

from .datastructures import Case, PromptFor, CallableExpression, get_all_possible_contexts, parse_string_to_expression


def prompt_user_for_expression(case: Union[Case, Table], prompt_for: PromptFor, target_name: str,
                               output_type: Type, session: Optional[Session] = None) -> Tuple[str, CallableExpression]:
    """
    Prompt the user for an executable python expression.

    :param case: The case to classify.
    :param prompt_for: The type of information ask user about.
    :param target_name: The name of the target attribute to compare the case with.
    :param output_type: The type of the output of the given statement from the user.
    :param session: The sqlalchemy orm session.
    :return: A callable expression that takes a case and executes user expression on it.
    """
    user_input, expression_tree = prompt_user_about_case(case, prompt_for, target_name)
    callable_expression = CallableExpression(user_input, output_type, expression_tree=expression_tree, session=session)
    return user_input, callable_expression


def prompt_user_about_case(case: Union[Case, Table], prompt_for: PromptFor, target_name: str) \
        -> Tuple[str, AST]:
    """
    Prompt the user for input.

    :param case: The case to prompt the user on.
    :param prompt_for: The type of information the user should provide for the given case.
    :param target_name: The name of the target property of the case that is queried.
    :return: The user input, and the executable expression that was parsed from the user input.
    """
    prompt_str = f"Give {prompt_for} for {case.__class__.__name__}.{target_name}"
    session = get_prompt_session_for_obj(case)
    user_input, expression_tree = prompt_user_input_and_parse_to_expression(prompt_str, session)
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
    completions += list(get_all_possible_contexts(obj).keys())
    return completions


def prompt_user_input_and_parse_to_expression(prompt: Optional[str] = None, session: Optional[PromptSession] = None,
                                              user_input: Optional[str] = None) -> Tuple[str, ast.AST]:
    """
    Prompt the user for input.

    :param prompt: The prompt to display to the user.
    :param session: The prompt session to use.
    :param user_input: The user input to use. If given, the user input will be used instead of prompting the user.
    :return: The user input and the AST tree.
    """
    while True:
        if not user_input:
            user_input = session.prompt(f"\n{prompt} >>> ")
        if user_input.lower() in ['exit', 'quit', '']:
            break
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
