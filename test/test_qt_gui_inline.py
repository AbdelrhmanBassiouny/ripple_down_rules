import inspect
from collections import UserDict

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QToolButton, QHBoxLayout
)
from PyQt6.QtCore import Qt
import sys

from colorama import Fore
from typing_extensions import Optional

from ripple_down_rules.datasets import load_zoo_dataset, Species
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.utils import is_iterable


class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QToolButton(checkable=True, checked=False)
        # self.toggle_button.setStyleSheet("""
        #     QToolButton {
        #         border: none;
        #         font-weight: bold;
        #         color: #FFA07A; /* Light orange */
        #     }
        # """)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.clicked.connect(self.toggle)

        self.title_label = QLabel(title)
        self.title_label.setTextFormat(Qt.TextFormat.RichText)  # Enable HTML rendering

        self.content_area = QWidget()
        self.content_area.setVisible(False)
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(15, 2, 0, 2)
        self.content_layout.setSpacing(2)

        layout = QVBoxLayout(self)
        header_layout = QHBoxLayout()
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.title_label)
        layout.addLayout(header_layout)
        layout.addWidget(self.content_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

    def toggle(self):
        is_expanded = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if is_expanded else Qt.ArrowType.RightArrow
        )
        self.content_area.setVisible(is_expanded)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


def python_colored_repr(value):
    if isinstance(value, str):
        return f'<span style="color:#90EE90;">"{value}"</span>'
    elif isinstance(value, (int, float)):
        return f'<span style="color:#ADD8E6;">{value}</span>'
    elif isinstance(value, bool) or value is None:
        return f'<span style="color:darkorange;">{value}</span>'
    elif isinstance(value, type):
        return f'<span style="color:#DCDCDC;">{{{value.__name__}}}</span>'
    elif callable(value):
        return ''
    else:
        return f'<span style="color:white;">{repr(value)}</span>'


class AttributeViewer(QWidget):
    def __init__(self, obj, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PyQt6 Inline Object Viewer with Syntax Coloring")
        self.setStyleSheet("background-color: #333333;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self.container_layout = QVBoxLayout(container)
        self.container_layout.setSpacing(2)
        self.container_layout.setContentsMargins(6, 6, 6, 6)

        self.add_attributes(obj, self.container_layout)

        scroll.setWidget(container)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def add_attributes(self, obj, layout, current_depth=0, max_depth=3):
        if current_depth > max_depth:
            return
        if isinstance(obj, dict):
            items = obj.items()
        elif isinstance(obj, (list, tuple, set)):
            items = enumerate(obj)
        else:
            methods = []
            attributes = []
            for attr in dir(obj):
                if attr.startswith("_") or attr == "scope":
                    continue
                try:
                    value = getattr(obj, attr)
                    if callable(value):
                        methods.append((attr, value))
                        continue
                except Exception as e:
                    value = f"<error: {e}>"
                attributes.append((attr, value))
            items = attributes + methods
        for attr, value in items:
            attr = f"{attr}"
            try:
                if is_iterable(value) or hasattr(value, "__dict__") and not inspect.isfunction(value):
                    self.add_collapsible(attr, value, layout, current_depth + 1, max_depth)
                else:
                    self.add_non_collapsible(attr, value, layout)
            except Exception as e:
                err = QLabel(f"<b>{attr}</b>: <span style='color:red;'>&lt;error: {e}&gt;</span>")
                err.setTextFormat(Qt.TextFormat.RichText)
                layout.addWidget(err)

    def add_collapsible(self, attr, value, layout, current_depth, max_depth):
        type_name = type(value) if not isinstance(value, type) else value
        collapsible = CollapsibleBox(f'<b><span style="color:#FFA07A;">{attr}</span> {python_colored_repr(type_name)}</b>')
        self.add_attributes(value, collapsible.content_layout, current_depth, max_depth)
        layout.addWidget(collapsible)

    def add_non_collapsible(self, attr, value, layout):
        type_name = type(value) if not isinstance(value, type) else value
        text = f'<b><span style="color:#FFA07A;">{attr}</span> {python_colored_repr(type_name)}</b>: {python_colored_repr(value)}'
        item_label = QLabel()
        item_label.setTextFormat(Qt.TextFormat.RichText)
        item_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        item_label.setStyleSheet("QLabel { padding: 1px; color: #FFA07A; }")
        item_label.setText(text)
        item_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(item_label)


# 🎯 Sample nested test object
class SubObject:
    def __init__(self):
        self.number = 42
        self.status = True
        self.message = "Hello from Sub"

class TestObject:
    def __init__(self):
        self.name = "Main"
        self.count = 3.14
        self.flag = False
        self.sub = SubObject()
        self.none_val = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    cases, targets = load_zoo_dataset(cache_file="zoo")
    cq = CaseQuery(cases[0], "species", (Species,), True, _target=targets[0])
    viewer = AttributeViewer(cq)
    viewer.resize(500, 600)
    viewer.show()
    sys.exit(app.exec())
