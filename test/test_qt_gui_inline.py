import inspect
from collections import UserDict

from PyQt6.QtGui import QPixmap, QPainter, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QToolButton, QHBoxLayout, QPushButton, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtCore import Qt
import sys

from colorama import Fore
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from typing_extensions import Optional

from ripple_down_rules.datasets import load_zoo_dataset, Species
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.utils import is_iterable


class ImageViewer(QGraphicsView):
    def __init__(self, image_path):
        super().__init__()
        self.setScene(QGraphicsScene(self))

        pixmap = QPixmap(image_path)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self._zoom = 0

    def wheelEvent(self, event):
        # Zoom in or out with Ctrl + mouse wheel
        if event.modifiers() == Qt.ControlModifier:
            angle = event.angleDelta().y()
            factor = 1.25 if angle > 0 else 0.8

            self._zoom += 1 if angle > 0 else -1
            if self._zoom > 10:  # max zoom in limit
                self._zoom = 10
                return
            if self._zoom < -10:  # max zoom out limit
                self._zoom = -10
                return

            self.scale(factor, factor)
        else:
            super().wheelEvent(event)



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
            # Calculate the vertical space used by buttons
            button_area_height = 0
            for i in range(self.layout.count()):
                item = self.layout.itemAt(i)
                if item.widget():
                    button_area_height += item.widget().height() + self.layout.spacing()

            remaining_height = self.height() - button_area_height
            if remaining_height <= 0:
                return  # No space to draw

            # Scale image to the remaining area (width, height)
            scaled = self.pixmap.scaled(
                self.width(),
                remaining_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            x = (self.width() - scaled.width()) // 2

            # Draw the image starting just below the buttons
            painter.drawPixmap(x, button_area_height+20, scaled)

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
        # self.adjust_size_recursive()

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

        self.setFixedSize(1600, 600)  # or your preferred initial size

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.setStyleSheet("background-color: #333333;")
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
        attr_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(attr_widget)

        self.add_collapsible(name, obj, self.attr_widget_layout, 0, 3)
        self.attr_widget_layout.addStretch()  # Push to top

        # === Right: Action buttons ===
        action_widget = BackgroundWidget('../images/thinking_pr2.jpg')

        ipython_console = IPythonConsole(locals())
        action_widget.layout.addWidget(ipython_console)

        self.viewer = ImageViewer("object_diagram.png")  # put your image path here

        # Add both to main layout
        main_layout.addWidget(scroll, stretch=1)
        main_layout.addWidget(action_widget, stretch=2)
        main_layout.addWidget(self.viewer, stretch=2)



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
            iterables = []
            for attr in dir(obj):
                if attr.startswith("_") or attr == "scope":
                    continue
                try:
                    value = getattr(obj, attr)
                    if callable(value):
                        methods.append((attr, value))
                        continue
                    elif is_iterable(value):
                        iterables.append((attr, value))
                        continue
                except Exception as e:
                    value = f"<error: {e}>"
                attributes.append((attr, value))
            items = attributes + iterables + methods
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


class IPythonConsole(RichJupyterWidget):
    def __init__(self, namespace=None, parent=None):
        super(IPythonConsole, self).__init__(parent)

        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt'
        self.command_log = []

        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        # Update the user namespace with your custom variables
        if namespace:
            self.kernel.shell.user_ns.update(namespace)

        # Set the underlying QTextEdit's palette
        palette = QPalette()
        self._control.setPalette(palette)

        # Override the stylesheet to force background and text colors
        self._control.setStyleSheet("""
                    background-color: #121212;
                    color: #00FF00;
                    selection-background-color: #006400;
                    selection-color: white;
                """)

        # Use a dark syntax style like monokai
        self.syntax_style = 'monokai'

        self.exit_requested.connect(self.stop)

    def execute(self, source=None, hidden=False, interactive=False):
        # Log the command before execution
        source = source if source is not None else self.input_buffer
        self.command_log.append(source)
        super().execute(source, hidden, interactive)

    def stop(self):
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()


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
    # viewer.resize(500, 600)
    viewer.show()
    sys.exit(app.exec())
