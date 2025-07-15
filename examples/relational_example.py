from relational_model import Part, Robot, PhysicalObject
from ripple_down_rules import GeneralRDR, CaseQuery


part_a = Part(name="A")
part_b = Part(name="B")
part_c = Part(name="C")
robot = Robot("pr2", parts=[part_a])
part_a.contained_objects = [part_b]
part_b.contained_objects = [part_c]

case_query = CaseQuery(robot, "contained_objects", (PhysicalObject,), False)

grdr = GeneralRDR(save_dir='./', model_name='part_containment_rdr')

grdr.fit_case(case_query)

result = grdr.classify(robot)
assert result['contained_objects'] == {part_b}