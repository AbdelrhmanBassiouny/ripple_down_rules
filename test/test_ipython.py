from IPython import start_ipython
from typing_extensions import Optional, List, Set


class PhysicalObject:
    def __init__(self, name: str):
        self.name = name
        self._contained_objects: Set[PhysicalObject] = set()

    @property
    def contained_objects(self):
        return self._contained_objects

    @contained_objects.setter
    def contained_objects(self, value):
        self._contained_objects = value


class Part(PhysicalObject):
    ...


class Robot(PhysicalObject):
    def __init__(self, name: str, parts: Optional[List[Part]] = None):
        super().__init__(name)
        self.parts = parts or []


part_a = Part(name="A")
part_b = Part(name="B")
part_c = Part(name="C")
part_d = Part(name="D")
robot = Robot("pr2", parts=[part_a, part_b, part_c, part_d])
part_a.contained_objects = {part_b, part_c}
part_c.contained_objects = {part_d}
case = robot

print("\nYou are now in an interactive shell. Type expressions like 'robot.speed == task.speed' to compare attributes.")
print("Type 'exit' or 'quit' to leave.")

# start_ipython(argv=[], user_ns=locals())
from IPython.core.inputtransformer2 import TransformerManager

transformer = TransformerManager()

while True:
    user_input = input("IPython command >>> ")
    if user_input.lower() in ['exit', 'quit']:
        break

    # Preprocess like IPython but don't execute
    transformed_input = transformer.transform_cell(user_input)
    print(f"Transformed input: {transformed_input}")
print(case.contained_objects)