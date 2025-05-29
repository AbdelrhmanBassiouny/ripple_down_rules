from __future__ import annotations

import os.path
from dataclasses import dataclass, field

from typing_extensions import List, Optional

from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.rdr import GeneralRDR


@dataclass(unsafe_hash=True)
class PhysicalObject:
    """
    A physical object is an object that can be contained in a container.
    """
    name: str
    contained_objects: List[PhysicalObject] = field(default_factory=list, hash=False)

@dataclass(unsafe_hash=True)
class Part(PhysicalObject):
    ...

@dataclass(unsafe_hash=True)
class Robot(PhysicalObject):
    parts: List[Part] = field(default_factory=list, hash=False)


part_a = Part(name="A")
part_b = Part(name="B")
part_c = Part(name="C")
robot = Robot("pr2", parts=[part_a])
part_a.contained_objects = [part_b]
part_b.contained_objects = [part_c]

case_query = CaseQuery(robot, "contained_objects", (PhysicalObject,), False)

load = True  # Set to True if you want to load an existing model, False if you want to create a new one.
if load and os.path.exists('./part_containment_rdr'):
    grdr = GeneralRDR.load('./', model_name='part_containment_rdr')
    grdr.ask_always = False # Set to True if you want to always ask the expert for a target value.
else:
    grdr = GeneralRDR(save_dir='./', model_name='part_containment_rdr')

grdr.fit_case(case_query)

print(grdr.classify(robot)['contained_objects'])
assert grdr.classify(robot)['contained_objects'] == {part_b}