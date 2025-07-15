# Tutorials for Ripple Down Rules

There are several ways to use Ripple Down Rules (RDR) in your projects. Below are some tutorials that demonstrate different use cases and how to implement them effectively.

## 1. Basic Usage of Ripple Down Rules

### Define your Data Model

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing_extensions import List


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
```

### Create your Case Object (Object to be Queried):

```python
part_a = Part(name="A")
part_b = Part(name="B")
part_c = Part(name="C")
robot = Robot("pr2", parts=[part_a])
part_a.contained_objects = [part_b]
part_b.contained_objects = [part_c]
```

### Define the Case Query
Here we define a query on the `robot` object to find out which objects are contained within it.
The output type is specified as `PhysicalObject`, and there can be multiple contained objects so we set `mutually_exclusive` to `False`.
```python
from ripple_down_rules import CaseQuery


case_query = CaseQuery(robot, "contained_objects", (PhysicalObject,), False)
```

### Load or Create the Ripple Down Rules Model

```python
from ripple_down_rules import GeneralRDR


grdr = GeneralRDR(save_dir='./', model_name='part_containment_rdr')
```

### Fit the Model using the Case Query
```python
grdr.fit_case(case_query)
```

### Classify the Object

```python
result = grdr.classify(robot)
assert result['contained_objects'] == {part_b}
```