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

        # # Add or update kernel metadata
        # if 'metadata' not in nb:
        #     nb['metadata'] = {}
        # nb.metadata['rdr_context'] = {
        #     'case_name': case_name,
        #     'function_name': func_name,
        #     'timestamp': time.time()
        # }
        # metadata for context
        nb.metadata.setdefault('rdr_context', {
            'case_name': case_name,
            'function_name': func_name,
            'timestamp': time.time()
        })


        # for cell in nb.cells:
        #     if 'case_name = ""' in cell.source:
        #         cell.source = cell.source.replace('case_name = ""', f'case_name = "{case_name}"')
        #
        #     # The placeholder function in the template is `my_rule_function`
        #     if cell.cell_type == 'code' and 'def my_rule_function(' in cell.source:
        #         cell.source = boilerplate_code
        #         # Update button to use the new function name
        #         for next_cell in nb.cells:
        #             if 'apply_rule_btn' in next_cell.source:
        #                 next_cell.source = next_cell.source.replace('my_rule_function', func_name)
        #                 break
        #         break
        # return nb
        # 1) Inject case name
        for cell in nb.cells:
            if isinstance(getattr(cell, 'source', None), str) and 'case_name = ""' in cell.source:
                cell.source = cell.source.replace('case_name = ""', f'case_name = "{case_name}"')
                break

        # 2) Replace the placeholder function cell with boilerplate code
        for cell in nb.cells:
            if getattr(cell, 'cell_type', '') == 'code' and isinstance(cell.source, str):
                if 'def my_rule_function(' in cell.source:
                    cell.source = boilerplate_code  # contains def {func_name}(...)
                    break

        # 3) Update only the buttons/toolbar cell with dynamic strings
        for cell in nb.cells:
            if getattr(cell, 'cell_type', '') == 'code' and isinstance(cell.source, str):
                if 'accept_rule_btn' in cell.source or 'toolbar' in cell.source or 'TARGET_FUNC_NAME' in cell.source:
                    src = cell.source
                    src = re.sub(r'COMM_FILE\s*=\s*[ru]?["\'].*?["\']',
                                 f'COMM_FILE = r"{self.communication_file}"',
                                 src)
                    src = re.sub(r'TARGET_FUNC_NAME\s*=\s*[ru]?["\'].*?["\']',
                                 f'TARGET_FUNC_NAME = "{func_name}"',
                                 src)
                    cell.source = src
                    break

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
                print("Jupyter process terminated by user.")
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


# import os
# import sys
# import subprocess
# import json
# import time
# import socket
# import uuid
# import threading
# import nbformat
# from jupyter_client import KernelManager
# from queue import Queue
# import re
#
#
# class JupyterNotebookManager:
#     """
#     Manages a Jupyter Notebook with direct kernel access for complex objects.
#     """
#
#     def __init__(self, template_path, print_cells=True):
#         self.template_path = template_path
#         self.print_cells = print_cells
#         self.notebook_process = None
#         self.notebook_url = None
#         self.notebook_port = None
#         self.notebook_path = None
#         self.kernel_id = None
#         self.kernel_manager = None
#         self.kernel_client = None
#         self.project_dir = os.path.dirname(os.path.abspath(__file__))
#         self.temp_dir = os.path.join(self.project_dir, "temp")
#         self.function_received = threading.Event()
#         self.received_function = None
#         self.direct_kernel_management = False
#
#         if not os.path.exists(self.temp_dir):
#             os.makedirs(self.temp_dir)
#
#     def enable_direct_kernel_management(self, enable=True):
#         """Enable or disable direct kernel management mode."""
#         self.direct_kernel_management = enable
#         return self
#
#     def create_notebook_with_direct_kernel_injection(self, case_obj, boilerplate_code, scope, imports=None):
#         """Create a notebook with direct kernel injection of complex objects."""
#         # Create modified notebook from template
#         notebook = nbformat.read(self.template_path, as_version=4)
#
#         # Modify cells to include boilerplate code and object references
#         for cell in notebook.cells:
#             if cell.cell_type == "code":
#                 # Replace function placeholder with boilerplate code
#                 if "def my_rule_function" in cell.source:
#                     cell.source = boilerplate_code
#
#                 # Replace object placeholder
#                 if "obj = None" in cell.source:
#                     # Don't modify the cell to assign directly - we'll handle this in the kernel
#                     cell.source = cell.source.replace("obj = None", "# obj is already initialized in the kernel")
#
#                 # Modify the send_cells_to_main_process function to use comm channel
#                 if "send_cells_to_main_process" in cell.source:
#                     notification_code = """
# def send_cells_to_main_process(btn):
#     print("Sending cell contents to main process...")
#     # Find the function cell
#     function_code = None
#     for i, cell_content in enumerate(get_ipython().user_ns.get('In', [])):
#         if cell_content and 'def my_rule_function' in cell_content:
#             function_code = cell_content
#             break
#
#     if function_code:
#         # Send function back through comm
#         try:
#             from ipykernel.comm import Comm
#             comm = get_ipython().kernel.comm_manager.get_comm('function_comm')
#             if comm:
#                 comm.send({'function': function_code})
#                 print("Function sent successfully through comm")
#             else:
#                 print("No comm channel available, writing to file")
#         except Exception as e:
#             print(f"Error sending function: {e}")
#     else:
#         print("Function not found in notebook cells")
#
#     # Apply function for visualization
#     my_rule_function(obj)
#     render_svg(obj)
# """
#                     cell.source = re.sub(r"def send_cells_to_main_process\(btn\):.*?render_svg\(obj\)",
#                                          notification_code.strip(), cell.source, flags=re.DOTALL)
#
#         # Write modified notebook to temporary file
#         notebook_id = str(uuid.uuid4())[:8]
#         self.notebook_path = os.path.join(self.temp_dir, f"notebook_{notebook_id}.ipynb")
#         nbformat.write(notebook, self.notebook_path)
#
#         return self.notebook_path
#
#     def _create_and_start_kernel(self):
#         """Create and start a kernel for direct management."""
#         try:
#             self.kernel_manager = KernelManager(kernel_name='python3')
#             self.kernel_manager.start_kernel()
#             self.kernel_id = self.kernel_manager.kernel_id
#
#             self.kernel_client = self.kernel_manager.client()
#             self.kernel_client.start_channels()
#             self.kernel_client.wait_for_ready(timeout=30)
#
#             return True
#         except Exception as e:
#             print(f"Failed to create and start kernel: {e}")
#             if self.kernel_client:
#                 self.kernel_client.stop_channels()
#             if self.kernel_manager:
#                 self.kernel_manager.shutdown_kernel()
#             return False
#
#     def inject_case_object_into_kernel(self, case_obj, scope=None):
#         """
#         Inject a case object into the kernel with improved class registration.
#
#         This method injects the object by ID reference and ensures the class
#         definition is properly loaded in the kernel context.
#         """
#         if not self.kernel_client:
#             print("Error: No kernel client available")
#             return False
#
#         # Initialize kernel object storage
#         self.kernel_client.execute("""
#         _kernel_objects = {}
#         """)
#
#         # Get complete class information
#         module_name = case_obj.__class__.__module__
#         class_name = case_obj.__class__.__name__
#         base_classes = [c.__module__ + '.' + c.__name__
#                         for c in case_obj.__class__.__mro__[1:]
#                         if c.__module__ != 'builtins']
#
#         # Import object's class and all base classes
#         self.kernel_client.execute(f"""
#         try:
#             import sys
#             import importlib
#
#             # Import main class
#             from {module_name} import {class_name}
#             print(f"Imported {class_name} from {module_name}")
#
#             # Import base classes
#             base_modules = {base_classes}
#             for base in base_modules:
#                 module_name, class_name = base.rsplit('.', 1)  # Variables defined inside this kernel code
#                 try:
#                     module = importlib.import_module(module_name)
#                     cls = getattr(module, class_name)
#                     print(f"Imported base class {{class_name}} from {{module_name}}")
#                 except Exception as import_err:  # Variable defined inside this kernel code
#                     print(f"Warning: Failed to import base class {{base}}: {{import_err}}")
#         except Exception as import_err:  # Variable defined inside this kernel code
#             print(f"Failed to import class: {{import_err}}")
#         """)
#
#         # Enhanced object receiver with better verification
#         self.kernel_client.execute("""
#         def _receive_object_reference(name, obj_id, expected_type):
#             import gc
#             import inspect
#
#             found_obj = None
#
#             # Find object by id in garbage collector
#             for obj in gc.get_objects():
#                 if id(obj) == obj_id:
#                     found_obj = obj
#                     break
#
#             if found_obj is None:
#                 print(f"No object found with ID: {obj_id}")
#                 return False
#
#             # Verify object type matches expected
#             actual_type = found_obj.__class__.__module__ + '.' + found_obj.__class__.__name__
#             if actual_type != expected_type:
#                 print(f"Type mismatch: Found {actual_type}, expected {expected_type}")
#                 # Continue anyway as this might be a false alarm
#
#             # Add to globals and verify callable status
#             globals()[name] = found_obj
#             _kernel_objects[name] = found_obj
#
#             # Debug information
#             print(f"Object '{name}' assigned with type: {type(found_obj).__name__}")
#             methods = [m for m in dir(found_obj) if callable(getattr(found_obj, m)) and not m.startswith('_')]
#             print(f"Object methods: {methods[:5]}{'...' if len(methods) > 5 else ''}")
#             print(f"Is callable: {callable(found_obj)}")
#
#             # Test direct method access
#             if len(methods) > 0:
#                 try:
#                     method_name = methods[0]
#                     method = getattr(found_obj, method_name)
#                     print(f"Method '{method_name}' exists and is {type(method)}")
#                 except Exception as method_err:
#                     print(f"Could not access method: {method_err}")
#
#             return True
#         """)
#
#         # Pass object ID with type verification
#         obj_id = id(case_obj)
#         expected_type = f"{module_name}.{class_name}"
#         result = self.kernel_client.execute_interactive(
#             f"_receive_object_reference('obj', {obj_id}, '{expected_type}')"
#         )
#
#         # Register object attributes for direct access
#         self.kernel_client.execute("""
#         if 'obj' in globals():
#             print("Setting up direct attribute access...")
#             try:
#                 # Create shortcut methods in global namespace
#                 methods = [m for m in dir(obj) if callable(getattr(obj, m)) and not m.startswith('_')]
#                 for method_name in methods[:10]:  # Limit to first 10 methods
#                     method = getattr(obj, method_name)
#                     globals()[f"obj_{method_name}"] = method
#                     print(f"Created shortcut for obj.{method_name}")
#             except Exception as attr_err:
#                 print(f"Error setting up attribute access: {attr_err}")
#         """)
#
#         # Add verification that object exists and is properly initialized
#         verify_result = self.kernel_client.execute_interactive("'obj' in globals() and obj is not None")
#         if "True" in str(verify_result):
#             print("Object successfully injected into kernel")
#             return True
#         else:
#             print("Failed to properly inject object into kernel")
#             return False
#
#     def run_notebook_server_with_direct_kernel(self):
#         """Run the notebook server using our pre-initialized kernel."""
#         if not self.kernel_manager:
#             raise RuntimeError("Kernel manager not initialized")
#
#         # Find available port
#         self.notebook_port = self._find_available_port()
#
#         # Setup command to start notebook with existing kernel
#         cmd = [
#             sys.executable, "-m", "jupyter", "notebook",
#             self.notebook_path,
#             "--no-browser",
#             f"--port={self.notebook_port}",
#             "--ip=127.0.0.1",
#             f"--NotebookApp.token=''",
#             f"--NotebookApp.password=''",
#             f"--existing={self.kernel_id}"
#         ]
#
#         # Start notebook server process
#         self.notebook_process = subprocess.Popen(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )
#
#         time.sleep(2)  # Wait for server to start
#
#         # Create the notebook URL
#         notebook_name = os.path.basename(self.notebook_path)
#         self.notebook_url = f"http://127.0.0.1:{self.notebook_port}/notebooks/{notebook_name}"
#
#         # Setup comm channel for receiving functions
#         self._setup_comm_channel()
#
#         print(f"Notebook server started at: {self.notebook_url}")
#         return {"url": self.notebook_url, "kernel_id": self.kernel_id}
#
#     def _setup_comm_channel(self):
#         """Set up a comm channel to receive function code from notebook."""
#         if not self.kernel_client:
#             return
#
#         try:
#             # Register comm target for receiving functions
#             setup_code = """
# from ipykernel.comm import Comm
# import json
#
# # Create comm for sending function back
# function_comm = Comm(target_name='function_comm')
#
# # Set up message handling
# def handle_comm_open(comm, msg):
#     print("Function comm opened")
#
#     @comm.on_msg
#     def _handle_msg(msg):
#         print("Message received through comm")
#
# get_ipython().kernel.comm_manager.register_target('function_comm', handle_comm_open)
# print("Function comm target registered")
# """
#             self.kernel_client.execute(setup_code)
#
#             # Set up handler for incoming messages
#             def handle_shell_msg(msg):
#                 msg_type = msg.get('msg_type', '')
#                 if msg_type == 'comm_msg':
#                     content = msg.get('content', {})
#                     data = content.get('data', {})
#                     if 'function' in data:
#                         self.received_function = data['function']
#                         self.function_received.set()
#
#             # Register message handler
#             self.kernel_client.shell_channel.connect()
#
#             def msg_handler(msg_list):
#                 for msg_raw in msg_list:
#                     try:
#                         msg = json.loads(msg_raw.decode('utf-8'))
#                         handle_shell_msg(msg)
#                     except Exception as e:
#                         print(f"Error handling message: {e}")
#
#             self.kernel_client.shell_channel.on_recv(msg_handler)
#
#         except Exception as e:
#             print(f"Error setting up comm channel: {e}")
#
#     def wait_for_button_press(self, timeout=300):
#         """Wait for the Accept Rule button to be pressed."""
#         # First wait for comm channel message
#         start_time = time.time()
#         while not self.function_received.is_set():
#             elapsed = time.time() - start_time
#             if timeout and elapsed > timeout:
#                 print(f"Timeout waiting for function submission after {elapsed:.1f} seconds")
#                 return None
#
#             # Check if notebook process is still running
#             if self.notebook_process and self.notebook_process.poll() is not None:
#                 print("Notebook process terminated unexpectedly")
#                 return None
#
#             time.sleep(0.5)
#
#         print("Function received through comm channel")
#         return {"function_cell": self.received_function}
#
#     def cleanup(self):
#         """Clean up all resources."""
#         # Stop kernel client
#         if self.kernel_client:
#             try:
#                 self.kernel_client.stop_channels()
#             except:
#                 pass
#             self.kernel_client = None
#
#         # Shutdown kernel
#         if self.kernel_manager:
#             try:
#                 self.kernel_manager.shutdown_kernel(now=True)
#             except:
#                 pass
#             self.kernel_manager = None
#
#         # Stop notebook process
#         if self.notebook_process:
#             try:
#                 self.notebook_process.terminate()
#                 self.notebook_process.wait(timeout=5)
#             except:
#                 try:
#                     self.notebook_process.kill()
#                 except:
#                     pass
#             self.notebook_process = None
#
#         # Delete temporary files
#         if self.notebook_path and os.path.exists(self.notebook_path):
#             try:
#                 os.remove(self.notebook_path)
#                 print(f"Removed temporary notebook file")
#             except:
#                 pass
#             self.notebook_path = None
#
#         print("All resources cleaned up")
#
#     def _find_available_port(self):
#         """Find an available port for the notebook server."""
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.bind(('', 0))
#         port = s.getsockname()[1]
#         s.close()
#         return port