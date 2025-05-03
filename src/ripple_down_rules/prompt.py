import ast
import logging
import os
import subprocess
import tempfile
from _ast import AST
from functools import cached_property

from IPython.core.magic import register_line_magic, line_magic, Magics, magics_class
from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config import Config
from typing_extensions import List, Optional, Tuple, Dict, Type

from .datastructures.enums import PromptFor
from .datastructures.case import Case
from .datastructures.callable_expression import CallableExpression, parse_string_to_expression
from .datastructures.dataclasses import CaseQuery
from .utils import extract_dependencies, contains_return_statement, make_set, get_imports_from_scope

@magics_class
class MyMagics(Magics):
    def __init__(self, shell, scope, output_type: Optional[Type] = None, func_name: str = "user_case",
                 func_doc: str = "User defined function to be executed on the case."):
        super().__init__(shell)
        self.scope = scope
        self.temp_file_path = None
        self.func_name = func_name
        self.func_doc = func_doc
        self.output_type = output_type
        self.user_edit_line = 0

    @line_magic
    def edit_case(self, line):

        boilerplate_code = self.build_boilerplate_code()

        self.write_to_file(boilerplate_code)

        print(f"Opening {self.temp_file_path} in PyCharm...")
        subprocess.Popen(["pycharm", "--line", str(self.user_edit_line), self.temp_file_path])

    def build_boilerplate_code(self):
        imports = self.get_imports()
        output_type_hint = f" -> {self.output_type.__name__}" if self.output_type else ""
        boilerplate = f"""{imports}\n\n
def {self.func_name}(case: {self.case_type.__name__}){output_type_hint}:
    \"\"\"{self.func_doc}\"\"\"
    # Write your code here
    pass
        """
        self.user_edit_line = imports.count('\n')+6
        return boilerplate

    def write_to_file(self, code: str):
        tmp = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".py")
        tmp.write(code)
        tmp.flush()
        self.temp_file_path = tmp.name
        tmp.close()

    def get_imports(self):
        case_type_import = f"from {self.case_type.__module__} import {self.case_type.__name__}"
        if self.output_type is None:
            output_type_import = f"from typing_extensions import Any"
        else:
            output_type_import = f"from {self.output_type.__module__} import {self.output_type.__name__}"
        imports = get_imports_from_scope(self.scope)
        imports = [i for i in imports if ("get_ipython" not in i)]
        if case_type_import not in imports:
            imports.append(case_type_import)
        if output_type_import not in imports:
            imports.append(output_type_import)
        imports = set(imports)
        return '\n'.join(imports)

    @cached_property
    def case_type(self) -> Type:
        """
        Get the type of the case object in the current scope.

        :return: The type of the case object.
        """
        case = self.scope['case']
        if isinstance(case, Case):
            return case._obj_type
        else:
            return type(case)

    @line_magic
    def load_case(self, line):
        if not self.temp_file_path:
            print("No file to load. Run %edit_case first.")
            return

        with open(self.temp_file_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == self.func_name:
                exec_globals = {}
                exec(source, self.scope, exec_globals)
                user_function = exec_globals[self.func_name]
                self.shell.user_ns[self.func_name] = user_function
                print(f"Loaded `{self.func_name}` function into user namespace.")
                return

        print(f"Function `{self.func_name}` not found.")


class CustomInteractiveShell(InteractiveShellEmbed):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        keys = ['output_type', 'func_name', 'func_doc']
        values = [kwargs.get(key, None) for key in keys]
        magics_kwargs = {key: value for key, value in zip(keys, values) if value is not None}
        self.register_magics(MyMagics(self, self.user_ns, **magics_kwargs))
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

    def __init__(self, scope: Optional[Dict] = None, header: Optional[str] = None,
                 output_type: Optional[Type] = None, prompt_for: Optional[PromptFor] = None):
        """
        Initialize the Ipython shell with the given scope and header.

        :param scope: The scope to use for the shell.
        :param header: The header to display when the shell is started.
        :param output_type: The type of the output from user input.
        :param prompt_for: The type of information to ask the user about.
        """
        self.scope: Dict = scope or {}
        self.header: str = header or ">>> Embedded Ipython Shell"
        self.output_type: Optional[Type] = output_type
        self.prompt_for: Optional[PromptFor] = prompt_for
        self.user_input: Optional[str] = None
        self.shell: CustomInteractiveShell = self._init_shell()
        self.all_code_lines: List[str] = []

    def _init_shell(self):
        """
        Initialize the Ipython shell with a custom configuration.
        """
        cfg = Config()
        func_name = None
        func_doc = self.header
        if self.prompt_for == PromptFor.Conditions:
            func_name = "get_conditions_for_case"
        elif self.prompt_for == PromptFor.Conclusion:
            func_name = "get_conclusion_for_case"
        shell = CustomInteractiveShell(config=cfg, user_ns=self.scope, banner1=self.header,
                                       output_type=self.output_type, func_name=func_name, func_doc=func_doc)
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
            if len(self.all_code_lines) == 1 and self.all_code_lines[0].strip() == '':
                self.user_input = None
            else:
                self.user_input = '\n'.join(self.all_code_lines)


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
    output_type = case_query.attribute_type if prompt_for == PromptFor.Conclusion else bool
    shell = IPythonShell(scope=scope, header=prompt_str, output_type=output_type, prompt_for=prompt_for)
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
