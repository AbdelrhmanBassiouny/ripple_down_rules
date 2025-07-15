from decorator_model import Part, Robot, PhysicalObject
from decorator_model import my_robot_factory
from ripple_down_rules import GeneralRDR, CaseQuery


# Define a simple robot with parts and containment relationships
robot = my_robot_factory()

# Optional: Use the GUI.
try:
    from PyQt6.QtWidgets import QApplication
    from ripple_down_rules.user_interface.gui import RDRCaseViewer
    app = QApplication([])
    viewer = RDRCaseViewer()
except ImportError:
    app = None
    viewer = None

# Create a GeneralRDR instance and fit it to the case query
robot.get_contained_objects()  # Ensure the decorator is applied

# Classify the robot to check if it contains part_b
result = robot.get_contained_objects()
assert result == robot.parts[0].contained_objects