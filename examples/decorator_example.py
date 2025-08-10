from decorator_model import Robot
from relational_model import Part, PhysicalObject
from ripple_down_rules.helpers import enable_gui


# Define a simple robot with parts and containment relationships
part_a = Part(name="A")
part_b = Part(name="B")
part_c = Part(name="C")
robot = Robot("pr2", parts=[part_a])
part_a.contained_objects = [part_b]
part_b.contained_objects = [part_c]

# Optional: Use the GUI.r
enable_gui()

# Create a GeneralRDR instance and fit it to the case query
robot.containment_rdr.fit = True
robot.get_contained_objects()

# Classify the robot to check if it contains part_b
robot.containment_rdr.fit = False
result = robot.get_contained_objects()
assert result == [part_b]
