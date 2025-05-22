import re
import sys
import inspect
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsTextItem, QGraphicsProxyWidget, QPushButton, QVBoxLayout,
    QWidget, QDialog, QLineEdit, QLabel, QDialogButtonBox, QScrollArea, QHBoxLayout
)
from PyQt6.QtGui import QBrush, QColor, QFontMetrics, QFont
from PyQt6.QtCore import QRectF
from PyQt6.QtCore import Qt


# from ripple_down_rules.rdr import GeneralRDR
# from test_helpers.helpers import get_fit_grdr

def is_custom_instance(obj):
    return hasattr(obj, '__dict__') and not type(obj).__module__ == 'builtins' and not inspect.isclass(obj)


import html

def format_python_literal(value):
    """Return HTML-colored string representation of the value."""
    s = repr(value)
    s = html.escape(s)

    # Highlight strings: single or double quoted
    s = re.sub(r'(&#x27;.*?&#x27;|".*?")', r"<span style='color:green'>\1</span>", s)

    # Highlight numbers
    s = re.sub(r'\b\d+(\.\d+)?\b', r"<span style='color:blue'>\g<0></span>", s)

    # Highlight True, False, None
    s = re.sub(r'\b(True|False|None)\b', r"<span style='color:darkorange'>\1</span>", s)

    return f"<code>{s}</code>"


class MethodInputDialog(QDialog):
    def __init__(self, method_name):
        super().__init__()
        self.setWindowTitle(f"Run Method: {method_name}")
        layout = QVBoxLayout()

        self.input_field = QLineEdit(self)
        layout.addWidget(QLabel("Enter arguments (comma-separated):"))
        layout.addWidget(self.input_field)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def get_input(self):
        return self.input_field.text()


class AttributeDialog(QDialog):
    def __init__(self, name, value):
        super().__init__()
        self.setWindowTitle(f"Attribute: {name}")
        layout = QVBoxLayout()

        layout.addWidget(QLabel(f"<b>Name:</b> {name}"))
        layout.addWidget(QLabel(f"<b>Type:</b> {type(value).__name__}"))

        value_label = QLabel(f"<b>Value:</b> {repr(value)}")
        value_label.setWordWrap(True)
        layout.addWidget(value_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        self.buttons.accepted.connect(self.accept)
        layout.addWidget(self.buttons)

        self.setLayout(layout)


class ClassBoxItem(QGraphicsRectItem):
    def __init__(self, obj, x=0, y=0, show_methods=True, viewer=None):
        super().__init__(x, y, 250, 150)
        self.obj = obj
        self.cls = obj if inspect.isclass(obj) else obj.__class__
        self.expanded = False
        self.show_methods = show_methods
        self.viewer = viewer  # Store viewer reference
        self.setBrush(QBrush(QColor("lightblue")))

        self.text_item = QGraphicsTextItem(self.cls.__name__, self)
        self.text_item.setPos(x + 10, y + 10)

        self.expand_button = QPushButton("Expand")
        self.expand_proxy = QGraphicsProxyWidget(self)
        self.expand_proxy.setWidget(self.expand_button)
        self.expand_proxy.setPos(x + 10, y + 40)
        self.expand_button.clicked.connect(self.show_scrollable_ui)

        self.scroll_proxy: Optional[QGraphicsProxyWidget] = None

        self.method_buttons = []
        self.attribute_buttons = []

    # Inside your ClassBoxItem class:
    def show_scrollable_ui(self):


        for btn in self.attribute_buttons:
            btn.setParent(None)
        self.attribute_buttons.clear()

        # remove scroll area from the scene
        if self.scroll_proxy:
            scroll_widget = self.scroll_proxy.widget()
            self.scroll_proxy.setWidget(None)  # Detach from proxy
            scroll_widget.deleteLater()  # Delete the actual QWidget
            self.scene().removeItem(self.scroll_proxy)
            self.scroll_proxy = None

        if self.expanded:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

            content = self.build_scrollable_content()
            scroll_area.setWidget(content)
            scroll_area.setFixedSize(int(self.rect().width()-100), int(self.rect().height()-100))  # Size of the "class box"

            self.scroll_proxy = QGraphicsProxyWidget(self)
            self.scroll_proxy.setWidget(scroll_area)
            self.scroll_proxy.setPos(self.rect().x() + 10, self.rect().y() + 90)  # Position inside your graphics scene
        else:
            self.expand_button.setText("Expand")

        self.expanded = not self.expanded

    def build_scrollable_content(self):

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_content.setLayout(scroll_layout)

        # Attributes
        i = 0
        for attr in dir(self.obj):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(self.obj, attr)
                if inspect.ismethod(value) or inspect.isfunction(value):
                    continue
            except Exception:
                value = "<unreadable>"
            full_repr = repr(value)
            font = QFont()
            metrics = QFontMetrics(font)
            elided_text = metrics.elidedText(full_repr, Qt.TextElideMode.ElideRight, 300)
            elided_html = format_python_literal(elided_text)

            # UI row
            row = QWidget()
            row_layout = QHBoxLayout()
            row.setLayout(row_layout)

            btn_str = f"Attr: {attr}"
            btn = QPushButton(btn_str)
            btn.clicked.connect(lambda _, a=attr: self.show_attribute(a))
            self.attribute_buttons.append(btn)
            label = QLabel(elided_html)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setStyleSheet("background-color: white; border: 1px solid #ccc; padding: 2px;")
            label.setToolTip(full_repr)
            self.attribute_buttons.append(label)

            row_layout.addWidget(btn)
            row_layout.addWidget(label)
            scroll_layout.addWidget(row)
            i += 1

        return scroll_content

    def toggle_expand(self):
        for btn in self.method_buttons:
            btn.setParent(None)
        self.method_buttons.clear()
        for btn in self.attribute_buttons:
            btn.setParent(None)
        self.attribute_buttons.clear()

        if not self.expanded:
            # Show attributes
            attributes = [a for a in dir(self.obj)
                          if not a.startswith("_") and not callable(getattr(self.obj, a))]
            # increase size of the box if the size of the attributes box is larger than the current size
            max_height = 150 + len(attributes) * 30
            # if the max height is larger than the window height, increase the height of the window
            if self.viewer and max_height > self.viewer.height():
                self.viewer.resize(self.viewer.width(), max_height + 100)
            self.setRect(self.rect().x(), self.rect().y(), self.rect().width(), max_height)
            for i, attr in enumerate(attributes):
                btn_str = f"Attr: {attr}"
                btn = QPushButton(btn_str)
                btn_proxy = QGraphicsProxyWidget(self)
                btn_proxy.setWidget(btn)
                byn_x = self.rect().x() + 10
                btn_y = self.rect().y() + 80 + i * 30
                btn_proxy.setPos(byn_x, btn_y)
                btn.clicked.connect(lambda _, a=attr: self.show_attribute(a))
                self.attribute_buttons.append(btn)

                # Create colored value label
                value = getattr(self.obj, attr, None)
                # Get full value as repr
                full_repr = repr(value)

                # Elide plain text to fit in a limited width (e.g. 200 px)
                font = QFont()
                metrics = QFontMetrics(font)
                elided_text = metrics.elidedText(full_repr, Qt.TextElideMode.ElideRight, 200)

                # Colorize only the elided version
                elided_html = format_python_literal(elided_text)

                val_label = QLabel(elided_html)
                val_label.setStyleSheet("background-color: white; padding: 2px; border: 1px solid #ccc;")
                val_label.setTextFormat(Qt.TextFormat.RichText)
                val_label.setToolTip(html.escape(full_repr))  # Full text in tooltip
                val_proxy = QGraphicsProxyWidget(self)
                val_proxy.setWidget(val_label)
                val_proxy.setPos(btn_proxy.pos().x() + btn.sizeHint().width() + 10, btn_y + 3)
                self.method_buttons.append(val_label)

            if self.show_methods:
                # Show methods below attributes
                methods = [m for m in dir(self.obj)
                           if not m.startswith("_")
                           and callable(getattr(self.obj, m))]
                for j, method in enumerate(methods[:10]):
                    btn = QPushButton(f"Run: {method}")
                    btn_proxy = QGraphicsProxyWidget(self)
                    btn_proxy.setWidget(btn)
                    y_offset = 80 + (len(attributes) + j) * 30
                    btn_proxy.setPos(self.rect().x() + 10, self.rect().y() + y_offset)
                    btn.clicked.connect(lambda _, m=method: self.run_method(m))
                    self.method_buttons.append(btn)

            self.expand_button.setText("Collapse")
        else:
            self.expand_button.setText("Expand")

        self.expanded = not self.expanded

    def run_method(self, method_name):
        dialog = MethodInputDialog(method_name)
        if dialog.exec():
            arg_str = dialog.get_input()
            try:
                args = eval(f"[{arg_str}]") if arg_str else []
                method = getattr(self.obj, method_name)
                result = method(*args)
                print(f"{method_name} result:", result)
            except Exception as e:
                print(f"Error running {method_name}: {e}")

    def show_attribute(self, attr_name):
        try:
            value = getattr(self.obj, attr_name)
            if is_custom_instance(value):
                # Open new UMLViewer window for the nested object
                viewer = UMLViewer(value, show_methods=False)
                viewer.setWindowTitle(f"Nested: {attr_name}")
                viewer.resize(1000, 1000)
                viewer.show()
                # Prevent garbage collection
                if self.viewer:
                    self.viewer.child_windows.append(viewer)
            else:
                dialog = AttributeDialog(attr_name, value)
                dialog.exec()
        except Exception as e:
            print(f"Failed to read attribute {attr_name}: {e}")


class UMLViewer(QGraphicsView):
    def __init__(self, obj, show_methods=True, parent_viewer=None):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.child_windows = []
        self.parent_viewer = parent_viewer  # Reference to owning viewer

        self.class_box = ClassBoxItem(obj, 50, 50, show_methods=show_methods, viewer=self)
        self.scene.addItem(self.class_box)

    def resize(self, width, height):
        super().resize(width, height)
        self.setSceneRect(QRectF(0, 0, width, height))
        self.scene.setSceneRect(QRectF(0, 0, width, height))
        self.class_box.setRect(50, 50, width - 100, height - 100)




# Example class to test
class MyClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"

    def add(self, x, y):
        return x + y


if __name__ == '__main__':
    app = QApplication(sys.argv)
    print(app)

    # cases, targets = load_zoo_dataset(cache_file="zoo")
    # cq = CaseQuery(cases[0], "species", (Species,), True , _target=targets[0])
    # grdr, _ = get_fit_grdr(cases, targets)
    viewer = UMLViewer(MyClass)
    viewer.setWindowTitle("UML Class Explorer")
    viewer.resize(1000, 1000)
    viewer.show()
    sys.exit(app.exec())
