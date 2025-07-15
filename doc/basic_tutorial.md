# Basic Usage of Ripple Down Rules

### Define your Data Model

Here we define a simple data model of a robot with parts both of which are physical objects and can contain other physical objects.
Put this in a file called `relational_model.py`:
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
In a new python script, create instances of the `Part` and `Robot` classes to represent your robot and its parts.
This will be the object that you will query with Ripple Down Rules.
```python
from relational_model import Part, Robot, PhysicalObject


part_a = Part(name="A")
part_b = Part(name="B")
part_c = Part(name="C")
robot = Robot("pr2", parts=[part_a])
part_a.contained_objects = [part_b]
part_b.contained_objects = [part_c]
```

### (Optional) Enable Ripple Down Rules GUI

If you want to use the GUI for Ripple Down Rules, ensure you have PyQt6 installed:
```bash
pip install pyqt6
sudo apt-get install libxcb-cursor-dev
```

Then, you can enable the GUI in your script as follows:
```python
# Enable GUI if available
try:
    from PyQt6.QtWidgets import QApplication
    from ripple_down_rules.user_interface.gui import RDRCaseViewer
    app = QApplication([])
    viewer = RDRCaseViewer()
except ImportError:
    app = None
    viewer = None
```

### Define the RDR Model and the Case Query
Here create/load our RDR model, then we define a query on the `robot` object to find out which objects are contained within it.
The output type is specified as `PhysicalObject`, and there can be multiple contained objects so we set `mutually_exclusive` to `False`.

Optionally enable the GUI.

```python
from ripple_down_rules import CaseQuery, GeneralRDR

# Enable GUI if available
try:
    from PyQt6.QtWidgets import QApplication
    from ripple_down_rules.user_interface.gui import RDRCaseViewer
    app = QApplication([])
    viewer = RDRCaseViewer()
except ImportError:
    app = None
    viewer = None

grdr = GeneralRDR(save_dir='./', model_name='part_containment_rdr')

case_query = CaseQuery(robot, "contained_objects", (PhysicalObject,), False)
```

### Fit the Model to the Case Query by Answering the prompts.
```python
grdr.fit_case(case_query)
```

### Finally, Classify the Object and Verify the Result

```python
result = grdr.classify(robot)
assert result['contained_objects'] == {part_b}
```