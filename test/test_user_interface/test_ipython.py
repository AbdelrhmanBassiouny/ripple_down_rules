from IPython.core.interactiveshell import ExecutionInfo
from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config import Config

from ripple_down_rules.utils import capture_variable_assignment


class IpythonShell:
    def __init__(self, scope=None, header=None):
        self.scope = scope or {}
        self.header = header or ">>> Embedded Ipython Shell"
        self.raw_condition = None
        self.shell = self._init_shell()
        self._register_hooks()

    def _init_shell(self):
        """
        Initialize the Ipython shell with a custom configuration.
        """
        cfg = Config()
        shell = InteractiveShellEmbed(config=cfg, user_ns=self.scope, banner1=self.header)
        return shell

    def _register_hooks(self):
        """
        Register hooks to capture specific events in the Ipython shell.
        """
        def capture_condition(exec_info: ExecutionInfo):
            code = exec_info.raw_cell
            if "condition" not in code:
                return
            # use ast to find if the user is assigning a value to the variable "condition"
            assignment = capture_variable_assignment(code, "condition")
            if assignment:
                # if the user is assigning a value to the variable "condition", update the raw_condition
                self.raw_condition = assignment
                print(f"[Captured Condition]:\n{self.raw_condition}")

        self.shell.events.register('pre_run_cell', capture_condition)

    def run(self):
        """
        Run the embedded shell.
        """
        self.shell()


def run_ipython_shell():
    x = 10
    msg = "hello"
    scope = locals()
    IpythonShell(scope).run()
    # Apply changes to outer scope
    x = scope['x']
    msg = scope['msg']
    print(f"Back in code: x={x}, msg={msg}")


# run_ipython_shell()
