import sys
import unittest

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
        cls.cases, cls.targets = load_zoo_dataset(cache_file="zoo")
        cls.cq = CaseQuery(cls.cases[0], "species", (Species,), True, _target=cls.targets[0])
        cls.viewer = RDRCaseViewer(cls.cq, "CaseQuery")
        cls.person = Person("Ahmed", Address("Cairo"))

    @classmethod
    def tearDownClass(cls):
        try:
            cls.viewer.show()
            sys.exit(cls.app.exec())
        except SystemExit:
            pass

    def test_change_title_text(self):
        self.viewer.title_label.setText(style("Changed Title", "o", 28, 'bold'))

    def test_update_image(self):
        self.viewer.obj_diagram_viewer.update_image("object_diagram_person.png")
        print("Image updated successfully.")

    def test_update_for_obj(self):
        self.viewer.update_for_object(self.person, "Person")
        print("Viewer updated successfully.")
