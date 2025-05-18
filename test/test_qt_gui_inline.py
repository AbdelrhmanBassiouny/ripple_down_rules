import inspect
from collections import UserDict

from PyQt6.QtGui import QPixmap, QPainter
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QToolButton, QHBoxLayout, QPushButton, QMainWindow
)
from PyQt6.QtCore import Qt
import sys

from colorama import Fore
from typing_extensions import Optional

from ripple_down_rules.datasets import load_zoo_dataset, Species
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.utils import is_iterable



class BackgroundWidget(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap(image_path)

        # Layout for buttons
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(10)

        accept_btn = QPushButton("Accept")
        accept_btn.setStyleSheet("background-color: #4CAF50; color: white;")  # Green button
        edit_btn = QPushButton("Edit")
        edit_btn.setStyleSheet("background-color: #2196F3; color: white;")  # Blue button

        self.layout.addWidget(accept_btn)
        self.layout.addWidget(edit_btn)
        self.layout.addStretch()  # Push buttons to top

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.pixmap.isNull():
            scaled = self.pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

    def resizeEvent(self, event):
        self.update()  # Force repaint on resize
        super().resizeEvent(event)


class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QToolButton(checkable=True, checked=False)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.clicked.connect(self.toggle)
        self.toggle_button.setStyleSheet("""
            QToolButton {
                border: none;
                font-weight: bold;
                color: #FFA07A; /* Light orange */
            }
        """)
        self.title_label = QLabel(title)
        self.title_label.setTextFormat(Qt.TextFormat.RichText)  # Enable HTML rendering
        self.title_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.title_label.setStyleSheet("QLabel { padding: 1px; color: #FFA07A; }")

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
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def toggle(self):
        is_expanded = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if is_expanded else Qt.ArrowType.RightArrow
        )
        self.content_area.setVisible(is_expanded)

        # Trigger resize
        self.adjust_size_recursive()

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def adjust_size_recursive(self):
        # Trigger resize
        self.adjustSize()

        # Traverse upwards to main window and call adjustSize on it too
        parent = self.parent()
        while parent:
            if isinstance(parent, QWidget):
                parent.layout().activate()  # Force layout refresh
                parent.adjustSize()
            elif isinstance(parent, QScrollArea):
                parent.widget().adjustSize()
                parent.viewport().update()
            if isinstance(parent, BackgroundWidget):
                parent.update()
                parent.updateGeometry()
                parent.repaint()
            if parent.parent() is None:
                top_window = parent.window()  # The main top-level window
                top_window.updateGeometry()
                top_window.repaint()
            parent = parent.parent()


def python_colored_repr(value):
    if isinstance(value, str):
        return f'<span style="color:#90EE90;">"{value}"</span>'
    elif isinstance(value, (int, float)):
        return f'<span style="color:#ADD8E6;">{value}</span>'
    elif isinstance(value, bool) or value is None:
        return f'<span style="color:darkorange;">{value}</span>'
    elif isinstance(value, type):
        return f'<span style="color:#C1BCBB;">{{{value.__name__}}}</span>'
    elif callable(value):
        return ''
    else:
        try:
            return f'<span style="color:white;">{repr(value)}</span>'
        except Exception as e:
            return f'<span style="color:red;">&lt;error: {e}&gt;</span>'


class AttributeViewer(QMainWindow):
    def __init__(self, obj, name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RDR Case Viewer")

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_widget.setStyleSheet("background-color: #333333;")


        main_layout = QHBoxLayout(main_widget)  # Horizontal layout to split window

        # === Left: Attributes ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        attr_widget = QWidget()
        self.attr_widget_layout = QVBoxLayout(attr_widget)
        self.attr_widget_layout.setSpacing(2)
        self.attr_widget_layout.setContentsMargins(6, 6, 6, 6)

        self.add_collapsible(name, obj, self.attr_widget_layout, 0, 3)
        self.attr_widget_layout.addStretch()  # Push to top

        scroll.setWidget(attr_widget)

        # === Right: Action buttons ===
        action_widget = BackgroundWidget('../images/thinking_pr2.jpg')

        # Add both to main layout
        main_layout.addWidget(attr_widget, stretch=1)
        main_layout.addWidget(action_widget, stretch=3)


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
        collapsible = CollapsibleBox(f'<b><span style="color:#FFA07A;">{attr}</span></b> {python_colored_repr(type_name)}')
        self.add_attributes(value, collapsible.content_layout, current_depth, max_depth)
        layout.addWidget(collapsible)

    def add_non_collapsible(self, attr, value, layout):
        type_name = type(value) if not isinstance(value, type) else value
        text = f'<b><span style="color:#FFA07A;">{attr}</span></b> {python_colored_repr(type_name)}: {python_colored_repr(value)}'
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
    viewer = AttributeViewer(cq, "CaseQuery")
    viewer.resize(500, 600)
    viewer.show()
    sys.exit(app.exec())
