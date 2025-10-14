import os
import re
import subprocess
import time
import json
import nbformat
from typing import Optional



class JupyterNotebookManager:
    """Manages the creation, execution, and cleanup of a Jupyter Notebook for user interaction."""

    def __init__(self, template_path: str, output_dir: str):
        self.template_path = template_path
        self.output_dir = output_dir
        self.notebook_path: Optional[str] = None
        self.communication_file: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None

    def create_and_run_notebook(self, case_name: str, boilerplate_code: str, func_name: str) -> Optional[str]:
        """Creates, runs a notebook, and waits for the function source code."""
        self.notebook_path = os.path.join(self.output_dir, "rdr_active_notebook.ipynb")
        self.communication_file = os.path.join(self.output_dir, ".rdr_notebook_comm.json")

        if os.path.exists(self.communication_file):
            os.remove(self.communication_file)

        notebook_content = self._create_notebook_content(case_name, boilerplate_code, func_name)
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

    def _create_notebook_content(self, case_name: str, boilerplate_code: str, func_name: str) -> nbformat.NotebookNode:
        """Creates the content of the notebook by modifying the template."""
        with open(self.template_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        # Add notebook-level metadata for init cell extension
        nb.metadata.setdefault('rdr_context', {
            'case_name': case_name,
            'function_name': func_name,
            'timestamp': time.time()
        })
        nb.metadata.setdefault('init_cell', {'run_on_load': True})

        for i, cell in enumerate(nb.cells):
            if isinstance(getattr(cell, 'source', None), str):
                # Function cell is the only one that should remain editable and not auto-run
                if getattr(cell, 'cell_type', '') == 'code' and 'def my_rule_function(' in cell.source:
                    cell.source = boilerplate_code
                    # Keep this cell visible and editable for user
                    cell.metadata = getattr(cell, 'metadata', {})
                    cell.metadata.update({
                        'editable': True,
                        'init_cell': False  # Don't auto-run
                    })
                # All other cells should be hidden but auto-run
                else:
                    # Update specific cells with their content
                    if 'case_name = ""' in cell.source:
                        cell.source = cell.source.replace('case_name = ""', f'case_name = "{case_name}"')
                    elif (
                            'accept_rule_btn' in cell.source or 'toolbar' in cell.source or 'TARGET_FUNC_NAME' in cell.source):
                        cell.source = re.sub(r'COMM_FILE\s*=\s*[ru]?["\'].*?["\']',
                                             f'COMM_FILE = r"{self.communication_file}"', cell.source)
                        cell.source = re.sub(r'TARGET_FUNC_NAME\s*=\s*[ru]?["\'].*?["\']',
                                             f'TARGET_FUNC_NAME = "{func_name}"', cell.source)

                    # Set up all non-function cells to be hidden but auto-run
                    cell.metadata = getattr(cell, 'metadata', {})
                    cell.metadata.update({
                        'tags': ['hide-input', 'init_cell'],
                        'jupyter': {'source_hidden': True},
                        'init_cell': True,  # Key for auto-execution
                        'trusted': True,
                        'editable': False,
                        'deletable': False
                    })

                    # Only hide outputs for non-visualization cells
                    if not ('render_svg' in cell.source and 'display(svg_output)' in cell.source):
                        cell.metadata['jupyter']['outputs_hidden'] = True
                        cell.metadata['tags'].append('hide-output')

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
                ['jupyter', 'notebook', self.notebook_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
            )
        except FileNotFoundError:
            print("Jupyter Notebook is not installed. Please install it with 'pip install notebook'.")
            raise

    def _wait_for_communication(self, timeout: int = 600) -> Optional[str]:
        """Waits for the communication file from the notebook or for the server to close."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                print("Jupyter process terminated.")
                return None
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
