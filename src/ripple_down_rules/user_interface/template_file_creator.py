import ast
import os
import shutil
import socket
import subprocess
import tempfile
from functools import cached_property
from textwrap import indent, dedent

from colorama import Fore, Style
from ipykernel.inprocess.ipkernel import InProcessInteractiveShell
from typing_extensions import Optional, Type, List, Callable

from ..datastructures.case import Case
from ..datastructures.dataclasses import CaseQuery
from ..datastructures.enums import Editor, PromptFor
from ..utils import str_to_snake_case, get_imports_from_scope, make_list, typing_hint_to_str, \
    get_imports_from_types, extract_function_or_class_source


def detect_available_editor() -> Optional[Editor]:
    """
    Detect the available editor on the system.

    :return: The first found editor that is available on the system.
    """
    editor_env = os.environ.get("RDR_EDITOR")
    if editor_env:
        return Editor.from_str(editor_env)
    for editor in [Editor.Pycharm, Editor.Code, Editor.CodeServer]:
        if shutil.which(editor.value):
            return editor
    return None


def is_port_in_use(port: int = 8080) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def start_code_server(workspace):
    """
    Start the code-server in the given workspace.
    """
    filename = os.path.join(os.path.dirname(__file__), "start-code-server.sh")
    os.system(f"chmod +x {filename}")
    print(f"Starting code-server at {filename}")
    return subprocess.Popen(["/bin/bash", filename, workspace], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


class TemplateFileCreator:
    """
    A class to create a rule template file for a given case and prompt for the user to edit it.
    """
    temp_file_path: Optional[str] = None
    """
    The path to the temporary file that is created for the user to edit.
    """
    port: int = int(os.environ.get("RDR_EDITOR_PORT", 8080))
    """
    The port to use for the code-server.
    """
    process: Optional[subprocess.Popen] = None
    """
    The process of the code-server.
    """
    all_code_lines: Optional[List[str]] = None
    """
    The list of all code lines in the function in the temporary file.
    """

    def __init__(self, shell: InProcessInteractiveShell, case_query: CaseQuery, prompt_for: PromptFor,
                 code_to_modify: Optional[str] = None, print_func: Optional[Callable[[str], None]] = None):
        self.print_func = print_func if print_func else print
        self.shell = shell
        self.code_to_modify = code_to_modify
        self.prompt_for = prompt_for
        self.case_query = case_query
        self.output_type = self.get_output_type()
        self.user_edit_line = 0
        self.func_name: str = self.get_func_name()
        self.func_doc: str = self.get_func_doc()
        self.function_signature: str = self.get_function_signature()
        self.editor: Optional[Editor] = detect_available_editor()
        self.workspace: str = os.environ.get("RDR_EDITOR_WORKSPACE", os.path.dirname(self.case_query.scope['__file__']))
        self.temp_file_path: str = os.path.join(self.workspace, "edit_code_here.py")

    def get_output_type(self) -> List[Type]:
        """
        :return: The output type of the function as a list of types.
        """
        if self.prompt_for == PromptFor.Conditions:
            output_type = bool
        else:
            output_type = self.case_query.attribute_type
        return make_list(output_type) if output_type is not None else None

    def edit(self):
        if self.editor is None:
            self.print_func(
                f"{Fore.RED}ERROR:: No editor found. Please install PyCharm, VSCode or code-server.{Style.RESET_ALL}")
            return

        boilerplate_code = self.build_boilerplate_code()
        self.write_to_file(boilerplate_code)

        self.open_file_in_editor()

    def open_file_in_editor(self):
        """
        Open the file in the available editor.
        """
        if self.editor == Editor.Pycharm:
            subprocess.Popen(["pycharm", "--line", str(self.user_edit_line), self.temp_file_path],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        elif self.editor == Editor.Code:
            subprocess.Popen(["code", self.temp_file_path])
        elif self.editor == Editor.CodeServer:
            try:
                subprocess.check_output(["pgrep", "-f", "code-server"])
                # check if same port is in use
                if is_port_in_use(self.port):
                    self.print_func(f"Code-server is already running on port {self.port}.")
                else:
                    raise ValueError("Port is not in use.")
            except (subprocess.CalledProcessError, ValueError) as e:
                self.process = start_code_server(self.workspace)
            self.print_func(f"Open code-server in your browser at http://localhost:{self.port}?folder={self.workspace}")
        self.print_func(f"Edit the file: {Fore.MAGENTA}{self.temp_file_path}")

    def build_boilerplate_code(self):
        imports = self.get_imports()
        if self.function_signature is None:
            self.function_signature = self.get_function_signature()
        if self.func_doc is None:
            self.func_doc = self.get_func_doc()
        if self.code_to_modify is not None:
            body = indent(dedent(self.code_to_modify), '    ')
        else:
            body = "    # Write your code here\n    pass"
        boilerplate = f"""{imports}\n\n{self.function_signature}\n    \"\"\"{self.func_doc}\"\"\"\n{body}"""
        self.user_edit_line = imports.count('\n') + 6
        return boilerplate

    def get_function_signature(self) -> str:
        if self.func_name is None:
            self.func_name = self.get_func_name()
        output_type_hint = self.get_output_type_hint()
        func_args = self.get_func_args()
        return f"def {self.func_name}({func_args}){output_type_hint}:"

    def get_output_type_hint(self) -> str:
        """
        :return: A string containing the output type hint for the function.
        """
        output_type_hint = ""
        if self.prompt_for == PromptFor.Conditions:
            output_type_hint = " -> bool"
        elif self.prompt_for == PromptFor.Conclusion:
            output_type_hint = f" -> {self.case_query.attribute_type_hint}"
        return output_type_hint

    def get_func_args(self) -> str:
        """
        :return: A string containing the function arguments.
        """
        if self.case_query.is_function:
            func_args = {}
            for k, v in self.case_query.case.items():
                if (self.case_query.function_args_type_hints is not None
                        and k in self.case_query.function_args_type_hints):
                    func_args[k] = typing_hint_to_str(self.case_query.function_args_type_hints[k])[0]
                else:
                    func_args[k] = type(v).__name__ if not isinstance(v, type) else f"Type[{v.__name__}]"
            func_args = ', '.join([f"{k}: {v}" if str(v) not in ["NoneType", "None"] else str(k)
                                   for k, v in func_args.items()])
        else:
            func_args = f"case: {self.case_type.__name__}"
        return func_args

    def write_to_file(self, code: str):
        if self.temp_file_path is None:
            tmp = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".py",
                                              dir=self.workspace)
            tmp.write(code)
            tmp.flush()
            self.temp_file_path = tmp.name
            tmp.close()
        else:
            with open(self.temp_file_path, 'w+') as f:
                f.write(code)

    def get_imports(self):
        """
        :return: A string containing the imports for the function.
        """
        case_type_imports = []
        if self.case_query.is_function:
            for k, v in self.case_query.case.items():
                if (self.case_query.function_args_type_hints is not None
                        and k in self.case_query.function_args_type_hints):
                    hint_list = typing_hint_to_str(self.case_query.function_args_type_hints[k])[1]
                    for hint in hint_list:
                        hint_split = hint.split('.')
                        if len(hint_split) > 1:
                            case_type_imports.append(f"from {'.'.join(hint_split[:-1])} import {hint_split[-1]}")
                else:
                    if isinstance(v, type):
                        case_type_imports.append(f"from {v.__module__} import {v.__name__}")
                    elif hasattr(v, "__module__") and not v.__module__.startswith("__"):
                        case_type_imports.append(f"\nfrom {type(v).__module__} import {type(v).__name__}")
        else:
            case_type_imports.append(f"from {self.case_type.__module__} import {self.case_type.__name__}")
        if self.output_type is None:
            output_type_imports = [f"from typing_extensions import Any"]
        else:
            output_type_imports = get_imports_from_types(self.output_type)
            if len(self.output_type) > 1:
                output_type_imports.append("from typing_extensions import Union")
            if list in self.output_type:
                output_type_imports.append("from typing_extensions import List")
        imports = get_imports_from_scope(self.case_query.scope)
        imports = [i for i in imports if ("get_ipython" not in i)]
        imports.extend(case_type_imports)
        imports.extend([oti for oti in output_type_imports if oti not in imports])
        imports = set(imports)
        return '\n'.join(imports)

    def get_func_doc(self) -> Optional[str]:
        """
        :return: A string containing the function docstring.
        """
        if self.prompt_for == PromptFor.Conditions:
            return (f"Get conditions on whether it's possible to conclude a value"
                    f" for {self.case_query.name}")
        else:
            return f"Get possible value(s) for {self.case_query.name}"

    def get_func_name(self) -> Optional[str]:
        func_name = ""
        if self.prompt_for == PromptFor.Conditions:
            func_name = f"{self.prompt_for.value.lower()}_for_"
        case_name = self.case_query.name.replace(".", "_")
        if self.case_query.is_function:
            # convert any CamelCase word into snake_case by adding _ before each capital letter
            case_name = case_name.replace(f"_{self.case_query.attribute_name}", "")
        func_name += case_name
        return str_to_snake_case(func_name)

    @cached_property
    def case_type(self) -> Type:
        """
        Get the type of the case object in the current scope.

        :return: The type of the case object.
        """
        case = self.case_query.scope['case']
        return case._obj_type if isinstance(case, Case) else type(case)

    def load(self) -> Optional[List[str]]:
        if not self.temp_file_path:
            self.print_func(f"{Fore.RED}ERROR:: No file to load. Run %edit first.{Style.RESET_ALL}")
            return None

        with open(self.temp_file_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)
        updates = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == self.func_name:
                exec_globals = {}
                exec(source, self.case_query.scope, exec_globals)
                user_function = exec_globals[self.func_name]
                updates[self.func_name] = user_function
                self.print_func(f"{Fore.BLUE}Loaded `{self.func_name}` function into user namespace.{Style.RESET_ALL}")
                break
        if updates:
            self.shell.user_ns.update(updates)
            self.all_code_lines = extract_function_or_class_source(self.temp_file_path,
                                                                   [self.func_name],
                                                                   join_lines=False)[self.func_name]
            return self.all_code_lines
        else:
            self.print_func(f"{Fore.RED}ERROR:: Function `{self.func_name}` not found.{Style.RESET_ALL}")
            return None

    def __del__(self):
        if hasattr(self, 'process') and self.process is not None and self.process.poll() is None:
            self.process.terminate()  # Graceful shutdown
            self.process.wait()  # Ensure cleanup
