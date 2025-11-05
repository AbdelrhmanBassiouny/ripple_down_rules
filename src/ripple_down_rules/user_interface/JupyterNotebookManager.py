import os
import subprocess
import time
import json
import nbformat
from typing import Optional, List, Callable
from jinja2 import Environment, FileSystemLoader

from .notebook_cells import (
    NotebookCell,
    create_standard_notebook_cells
)


class JupyterNotebookManager:
    """Manages the creation, execution, and cleanup of a Jupyter Notebook for user interaction."""

    def __init__(self, template_dir: str, output_dir: str):
        """
        Initialize the notebook manager.

        Args:
            template_dir: Directory containing Jinja2 templates
            output_dir: Directory where notebooks and communication files are stored
        """
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.notebook_path: Optional[str] = None
        self.communication_file: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def run_workflow_from_case_query(
        self,
        case_query,  # CaseQuery type
        prompt_for,  # PromptFor type
        rdr_instance,
        print_func: Callable[[str], None] = print
    ) -> Optional[str]:
        """
        Execute complete notebook workflow from case query.

        Args:
            case_query: The case query to prompt for
            prompt_for: Type of prompt (Conditions/Conclusion)
            rdr_instance: RDR instance for visualization
            print_func: Function for console output

        Returns:
            Formatted function source code, or None if cancelled
        """
        # Import here to avoid circular dependencies
        from .template_file_creator import TemplateFileCreator

        # Create template from case_query
        template = TemplateFileCreator(case_query, prompt_for)

        # Generate SVG if RDR instance has visualization
        svg_file_path = None
        if rdr_instance and hasattr(rdr_instance, 'rdr_dot') and rdr_instance.rdr_dot:
            try:
                user_interface_dir = os.path.dirname(__file__)
                case_query.render_rule_tree(
                    os.path.join(user_interface_dir, "rule_tree"),
                    view=False
                )
                svg_file_path = os.path.join(user_interface_dir, "rule_tree.svg")
                if not os.path.exists(svg_file_path):
                    svg_file_path = None
            except Exception as e:
                print_func(f"Warning: Could not generate rule tree: {e}")

        # Build boilerplate code
        boilerplate_code = template.build_boilerplate_code()
        func_name = template.func_name

        # Get case name
        from ..utils import get_full_class_name
        case_name = get_full_class_name(case_query.case_type)

        template.create_case_in_database()

        # Create and run notebook
        raw_source = self.create_and_run_notebook(
            case_name=case_name,
            boilerplate_code=boilerplate_code,
            func_name=func_name,
            svg_file_path=svg_file_path
        )

        # Return None if cancelled
        if raw_source is None:
            return None

        # Parse and format source code
        try:
            all_code_lines, updates = TemplateFileCreator.load_from_source(
                raw_source, func_name, print_func
            )
            if all_code_lines is None:
                return None
            return '\n'.join(all_code_lines)
        except Exception as e:
            print_func(f"Error parsing function: {e}")
            raise

    def create_and_run_notebook(
        self,
        case_name: str,
        boilerplate_code: str,
        func_name: str,
        svg_file_path: Optional[str] = None,
        connection_string: str = "rdr@localhost:3306/RDR"
    ) -> Optional[str]:
        """
        Creates, runs a notebook, and waits for the function source code.

        Args:
            case_name: Full qualified class name (e.g., "ripple_down_rules.datasets.Robot")
            boilerplate_code: Initial Python function code for the user to edit
            func_name: Name of the function to be edited
            svg_data: Optional SVG data for rule tree visualization
            connection_string: Database connection string

        Returns:
            Function source code as string, or None if interaction failed
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self.notebook_path = os.path.join(self.output_dir, "rdr_active_notebook.ipynb")
        self.communication_file = os.path.join(self.output_dir, ".rdr_notebook_comm.json")

        if os.path.exists(self.communication_file):
            os.remove(self.communication_file)

        # Create cells using the component architecture
        cells = create_standard_notebook_cells(
            case_name=case_name,
            func_name=func_name,
            boilerplate_code=boilerplate_code,
            comm_file=self.communication_file,
            connection_string=connection_string,
            svg_file_path=svg_file_path
        )

        # Build notebook from cells
        notebook_content = self.create_notebook_from_cells(
            cells=cells,
            case_name=case_name,
            func_name=func_name
        )

        with open(self.notebook_path, 'w') as f:
            nbformat.write(notebook_content, f)

        self._start_jupyter()

        try:
            function_source = self._wait_for_communication()
            return function_source
        except (FileNotFoundError, KeyboardInterrupt) as e:
            print(f"Notebook interaction was interrupted or failed: {e}")
            return None
        finally:
            self._cleanup()

    def create_notebook_from_cells(
        self,
        cells: List[NotebookCell],
        case_name: str,
        func_name: str
    ) -> nbformat.NotebookNode:
        """
        Build a notebook from cell components.

        Args:
            cells: List of NotebookCell instances
            case_name: Full qualified class name
            func_name: Name of the function being edited

        Returns:
            nbformat.NotebookNode ready to be written to disk
        """
        # Create notebook structure
        nb = nbformat.v4.new_notebook()

        # Add cells
        for cell_component in cells:
            cell_dict = cell_component.to_nbformat_dict()
            if cell_component.cell_type.value == 'code':
                cell = nbformat.v4.new_code_cell(
                    source=cell_dict['source'],
                    metadata=cell_dict['metadata']
                )
            else:
                cell = nbformat.v4.new_markdown_cell(
                    source=cell_dict['source'],
                    metadata=cell_dict['metadata']
                )
            nb.cells.append(cell)

        # Set notebook metadata
        nb.metadata.update({
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {
                    'name': 'ipython',
                    'version': 3
                },
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.8.0'
            },
            'rdr_context': {
                'case_name': case_name,
                'function_name': func_name,
                'timestamp': time.time()
            },
            'init_cell': {
                'run_on_load': True
            }
        })

        return nb

    def _start_jupyter(self):
        """Starts the Jupyter Notebook server and opens the notebook."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))

            env = os.environ.copy()
            python_path = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = f"{project_root}:{python_path}" if python_path else project_root

            self.process = subprocess.Popen(
                # ['jupyter', 'notebook', self.notebook_path],
                  ['pycharm', self.notebook_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
            )
        except FileNotFoundError:
            print("Jupyter Notebook is not installed. Please install it with 'pip install notebook'.")
            raise

    def _wait_for_communication(self, timeout: int = 600) -> Optional[str]:
        """Waits for the communication file from the notebook or for the server to close."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # if self.process and self.process.poll() is not None:
            #     print("Jupyter process terminated.")
            #     return None
            if os.path.exists(self.communication_file):
                with open(self.communication_file, 'r') as f:
                    data = json.load(f)
                    return data.get('function_source')
            time.sleep(1)
        raise FileNotFoundError(f"Communication file not found after {timeout} seconds.")

    def _cleanup(self):
        """Cleans up resources like the notebook server process and temporary files."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()

        if self.notebook_path and os.path.exists(self.notebook_path):
            os.remove(self.notebook_path)
        if self.communication_file and os.path.exists(self.communication_file):
            os.remove(self.communication_file)
