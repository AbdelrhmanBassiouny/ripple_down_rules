from relational_model import Part, Robot, PhysicalObject
from ripple_down_rules import GeneralRDR, CaseQuery
from ripple_down_rules.helpers import enable_gui

# Define a simple robot with parts and containment relationships
part_a = Part(name="A")
part_b = Part(name="B")
part_c = Part(name="C")
robot = Robot("pr2", parts=[part_a])
part_a.contained_objects = [part_b]
part_b.contained_objects = [part_c]

# Optional: Use the GUI.
enable_gui()

# Create a GeneralRDR instance and fit it to the case query
grdr = GeneralRDR(save_dir='./', model_name='part_containment_rdr')

case_query = CaseQuery(robot, "contained_objects", (PhysicalObject,), False)

grdr.fit_case(case_query)

# Classify the robot to check if it rcontains part_b
result = grdr.classify(robot)
assert list(result['contained_objects']) == [part_b]