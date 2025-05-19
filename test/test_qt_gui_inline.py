import sys
import unittest

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication
)
from anyio import sleep
from typing_extensions import List

from ripple_down_rules.datasets import load_zoo_dataset, Species
from ripple_down_rules.datastructures.case import Case
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.gui import RDRCaseViewer, style
from test_object_diagram import Person, Address


@unittest.skip("GUI tests need visual inspection and cannot be run automatically.")
class GUITestCase(unittest.TestCase):
    """Test case for the GUI components of the ripple down rules package."""
    app: QApplication
    viewer: RDRCaseViewer
    cq: CaseQuery
    cases: List[Case]
    person: Person

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)
        cls.cases, cls.targets = load_zoo_dataset(cache_file="./test_results/zoo")
        cls.cq = CaseQuery(cls.cases[0], "species", (Species,), True, _target=cls.targets[0])
        cls.viewer = RDRCaseViewer()
        cls.person = Person("Ahmed", Address("Cairo"))

    def test_change_title_text(self):
        self.viewer.show()
        self.app.exec()
        self.viewer.title_label.setText(style("Changed Title", "o", 28, 'bold'))
        self.viewer.show()
        self.app.exec()

    def test_update_image(self):
        self.viewer.obj_diagram_viewer.update_image("./test_helpers/object_diagram_case_query.png")
        self.viewer.show()
        self.app.exec()
        self.viewer.obj_diagram_viewer.update_image("./test_helpers/object_diagram_person.png")
        self.viewer.show()
        self.app.exec()

    def test_update_for_obj(self):
        self.viewer.update_for_object(self.cq, "CaseQuery")
        self.viewer.show()
        self.app.exec()
        self.viewer.update_for_object(self.person, "Person")
        self.viewer.show()
        self.app.exec()
