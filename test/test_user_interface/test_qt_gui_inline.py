import os.path
import unittest

try:
    from PyQt6.QtWidgets import QApplication
except ImportError as e:
    print(f"Skipping GUI tests due to missing PyQt6: {e}")
    QApplication = None
from typing_extensions import List

from ripple_down_rules.datasets import load_zoo_dataset, Species
from ripple_down_rules.datastructures.case import Case
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.user_interface.gui import RDRCaseViewer, style
from test_helpers.helpers import get_fit_grdr
from test_object_diagram import Person, Address


@unittest.skipIf(QApplication is None, "GUI tests need PyQt6,"
                                       "and they need visual inspection by a user and cannot be run automatically.")
class GUITestCase(unittest.TestCase):
    """Test case for the GUI components of the ripple down rules package."""
    app: QApplication
    viewer: RDRCaseViewer
    cq: CaseQuery
    cases: List[Case]
    person: Person

    @classmethod
    def setUpClass(cls):
        print("Setting up GUI test case...")
        cls.app = QApplication([])
        cls.cases, cls.targets = load_zoo_dataset(cache_file=f"{os.path.dirname(__file__)}/../test_results/zoo")
        cls.cq = CaseQuery(cls.cases[0], "species", (Species,), True, _target=cls.targets[0])
        cls.viewer = RDRCaseViewer(save_file=f"{os.path.dirname(__file__)}/../test_results/grdr_viewer")
        cls.person = Person("Ahmed", Address("Cairo"))

    def test_change_title_text(self):
        self.viewer.show()
        self.app.exec()
        self.viewer.title_label.setText(style("Changed Title", "o", 28, 'bold'))
        self.viewer.show()
        self.app.exec()

    def test_update_image(self):
        self.viewer.obj_diagram_viewer.update_image(f"{os.path.dirname(__file__)}/../test_helpers/object_diagram_case_query.png")
        self.viewer.show()
        self.app.exec()
        self.viewer.obj_diagram_viewer.update_image(f"{os.path.dirname(__file__)}/../test_helpers/object_diagram_person.png")
        self.viewer.show()
        self.app.exec()

    def test_update_for_obj(self):
        self.viewer.update_for_object(self.cq, "CaseQuery")
        self.viewer.show()
        self.app.exec()
        self.viewer.update_for_object(self.person, "Person")
        self.viewer.show()
        self.app.exec()

    def test_save_button(self):
        grdr, _ = get_fit_grdr(self.cases, self.targets)
        grdr.set_viewer(self.viewer)
        self.viewer.title_label.setText(style("Press `Save` To Test", "o", 28, 'bold'))
        self.viewer.show()
        self.app.exec()
        grdr_viewer_dir = f"{os.path.dirname(__file__)}/../test_results/grdr_viewer"
        self.assertTrue(os.path.exists(f"{grdr_viewer_dir}/{grdr.model_name}/rdr_metadata/{grdr.model_name}.json"))

