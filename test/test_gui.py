import sys
import inspect
import threading

from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QVBoxLayout,
    QWidget, QListWidget, QLabel, QSplitter, QTreeWidgetItem
)

from prompt_toolkit import PromptSession
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers import PythonLexer
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.patch_stdout import patch_stdout

class DebuggerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyDebugger GUI")
        self.setGeometry(100, 100, 1000, 700)

        self.stack = []
        self.current_frame_index = 0

        self.init_ui()
        self.start_debugger()

        # Run REPL in background
        threading.Thread(target=self.run_repl, daemon=True).start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        splitter = QSplitter()

        # Stack viewer
        self.stack_list = QListWidget()
        self.stack_list.currentRowChanged.connect(self.change_frame)
        splitter.addWidget(self.stack_list)

        # Variable viewer
        from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem
        self.vars_tree = QTreeWidget()
        self.vars_tree.setHeaderLabels(["Name", "Value"])
        splitter.addWidget(self.vars_tree)

        splitter.setSizes([200, 800])
        layout.addWidget(splitter)

        # Output / logs
        self.output_label = QLabel("REPL running in terminal. Output here is optional.")
        layout.addWidget(self.output_label)

        central_widget.setLayout(layout)

        # self.setStyleSheet("""
        #     QListWidget, QTextEdit, QLabel {
        #         color: #f0f0f0;
        #         background-color: #2b2b2b;
        #         font-family: Consolas, monospace;
        #         font-size: 14px;
        #     }
        #     QListWidget::item:selected {
        #         background-color: #3c3f41;
        #     }
        #     QWidget {
        #         background-color: #2b2b2b;
        #     }
        # """)

    def start_debugger(self):
        # Simulate getting stack (frame where called)
        frame = inspect.currentframe().f_back.f_back  # go up to caller of outer()
        self.stack = []
        while frame:
            self.stack.append(frame)
            frame = frame.f_back
        self.stack.reverse()
        self.current_frame_index = len(self.stack) - 1
        self.update_stack_view()

    def update_stack_view(self):
        self.stack_list.blockSignals(True)  # prevent triggering change_frame
        self.stack_list.clear()
        for i, f in enumerate(self.stack):
            label = f"{f.f_code.co_name} (line {f.f_lineno})"
            if i == self.current_frame_index:
                label = "-> " + label
            self.stack_list.addItem(label)
        self.stack_list.setCurrentRow(self.current_frame_index)
        self.stack_list.blockSignals(False)  # re-enable signal
        self.update_vars_view()

    def update_vars_view(self):
        def add_variable_item(parent, name, value, visited=None):
            if visited is None:
                visited = set()

            # Color logic
            def colored_value(val):
                val_str = repr(val)
                color = QColor("#cccccc")  # default
                if isinstance(val, str):
                    color = QColor("#6a8759")  # greenish
                elif isinstance(val, (int, float)):
                    color = QColor("#6897bb")  # blue
                elif isinstance(val, bool):
                    color = QColor("#cc7832")  # orange
                elif val is None:
                    color = QColor("#cc7832")  # orange
                elif isinstance(val, (dict, list, tuple, set)):
                    color = QColor("#9876aa")  # purple
                elif hasattr(val, '__class__'):
                    color = QColor("#ffc66d")  # class name
                return val_str, color

            val_str, color = colored_value(value)
            item = QTreeWidgetItem([str(name), val_str])
            item.setForeground(1, QBrush(color))  # Set value color
            item.setForeground(0, QBrush(QColor("#a9b7c6")))  # name color
            parent.addChild(item)

            # Prevent circular refs
            if id(value) in visited:
                return
            visited.add(id(value))

            # Recurse
            if isinstance(value, dict):
                for k, v in value.items():
                    add_variable_item(item, f"{k!r}", v, visited)
            elif isinstance(value, (list, tuple, set)):
                for i, v in enumerate(value):
                    add_variable_item(item, f"[{i}]", v, visited)
            elif hasattr(value, '__dict__'):
                for attr, val in vars(value).items():
                    add_variable_item(item, attr, val, visited)

        self.vars_tree.clear()
        frame = self.stack[self.current_frame_index]

        root = QTreeWidgetItem(["[Locals]", ""])
        self.vars_tree.addTopLevelItem(root)
        for k, v in frame.f_locals.items():
            add_variable_item(root, k, v)

        root_globals = QTreeWidgetItem(["[Globals]", ""])
        self.vars_tree.addTopLevelItem(root_globals)
        for k, v in list(frame.f_globals.items())[:10]:  # limit for now
            add_variable_item(root_globals, k, v)
        self.vars_tree.expandAll()

    def change_frame(self, index):
        if 0 <= index < len(self.stack):
            self.current_frame_index = index
            self.update_stack_view()

    def run_repl(self):
        completer = WordCompleter(['print', 'len', 'locals', 'globals', 'up', 'down', 'exit'], ignore_case=True)
        session = PromptSession(lexer=PygmentsLexer(PythonLexer), completer=completer, multiline=True)

        with patch_stdout():
            while True:
                try:
                    text = session.prompt(">>> ")
                    if text.strip() == "exit":
                        break
                    elif text.strip() == "up":
                        self.current_frame_index = max(0, self.current_frame_index - 1)
                        self.update_stack_view()
                        continue
                    elif text.strip() == "down":
                        self.current_frame_index = min(len(self.stack) - 1, self.current_frame_index + 1)
                        self.update_stack_view()
                        continue

                    frame = self.stack[self.current_frame_index]

                    try:
                        result = eval(text, frame.f_globals, frame.f_locals)
                        print(result)
                    except Exception:
                        exec(text, frame.f_globals, frame.f_locals)
                    self.update_vars_view()

                except Exception as e:
                    print(f"REPL error: {e}")

class Sample:
    def __init__(self):
        self.name = "debugger"
        self.child = {"x": 1, "y": [1, 2, 3]}

def test():
    obj = Sample()
    nested = {"a": 123, "b": [1, 2, {"deep": obj}]}
    inner()

def inner():
    x = [1, 2, 3]
    y = {"key": "value"}
    app = QApplication(sys.argv)
    win = DebuggerWindow()
    win.show()
    app.exec_()

test()
