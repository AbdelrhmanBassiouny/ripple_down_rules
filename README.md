# Ripple Down Rules (RDR)

A python implementation of the various ripple down rules versions, including Single Classification (SCRDR),
Multi Classification (MCRDR), and Generalised Ripple Down Rules (GRDR).

SCRDR, MCRDR, and GRDR are rule-based classifiers that are built incrementally, and can be used to classify
data cases. The rules are refined as new data cases are classified.

SCRDR, MCRDR, and GRDR implementation were inspired from the book:
["Ripple Down Rules: An Alternative to Machine Learning"](https://doi.org/10.1201/9781003126157) by Paul Compton, Byeong Ho Kang.

## Installation

```bash
sudo apt-get install graphviz graphviz-dev
pip install ripple_down_rules
```
For GUI support, also install:

```bash
sudo apt-get install libxcb-cursor-dev
```

## Example Usage

### Propositional Example

By propositional, I mean that each rule conclusion is a propositional logic statement with a constant value.

For this example, we will use the [UCI Zoo dataset](https://archive.ics.uci.edu/ml/datasets/zoo) to classify animals
into their species based on their features. The dataset contains 101 animals with 16 features, and the target is th
e species of the animal.

To install the dataset:
```bash
pip install ucimlrepo
```

```python
from __future__ import annotations
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.datastructures.case import create_cases_from_dataframe
from ripple_down_rules.rdr import GeneralRDR
from ripple_down_rules.utils import render_tree
from ucimlrepo import fetch_ucirepo
from enum import Enum

class Species(str, Enum):
    """Enum for the species of the animals in the UCI Zoo dataset."""
    mammal = "mammal"
    bird = "bird"
    reptile = "reptile"
    fish = "fish"
    amphibian = "amphibian"
    insect = "insect"
    molusc = "molusc"
    
    @classmethod
    def from_str(cls, value: str) -> Species:
        return getattr(cls, value)

# fetch dataset
zoo = fetch_ucirepo(id=111)

# data (as pandas dataframes)
X = zoo.data.features
y = zoo.data.targets

# This is a utility that allows each row to be a Case instance,
# which simplifies access to column values using dot notation.
all_cases = create_cases_from_dataframe(X, name="Animal")

# The targets are the species of the animals
category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
targets = [Species.from_str(category_id_to_name[i]) for i in y.values.flatten()]

# Now that we are done with the data preparation, we can create and use the Ripple Down Rules classifier.
grdr = GeneralRDR()

# Fit the GRDR to the data
case_queries = [CaseQuery(case, 'species', type(target), True, _target=target)
                for case, target in zip(all_cases[:10], targets[:10])]
grdr.fit(case_queries, animate_tree=True)

# Render the tree to a file
render_tree(grdr.start_rules[0], use_dot_exporter=True, filename="species_rdr")

# Classify a case
cat = grdr.classify(all_cases[50])['species']
assert cat == targets[50]
```

### Relational Example

By relational, I mean that each rule conclusion is not a constant value, but is related to the case being classified,
you can understand it better by the next example.

In this example, we will create a simple robot with parts and use Ripple Down Rules to find the contained objects inside
another object, in this case, a robot. You see, the result of such a rule will vary depending on the robot 
and the parts it has.

```python
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
```

When prompted to write a rule, I wrote the following inside the template function that the Ripple Down Rules created
for me, this function takes a `case` object as input:

```python
contained_objects = []
for part in case.parts:
    contained_objects.extend(part.contained_objects)
return contained_objects
```

And then when asked for conditions, I wrote the following inside the template function that the Ripple Down Rules
created:

```python
return len(case.parts) > 0
```

This means that the rule will only be applied if the robot has parts.

If you notice, the result only contains part B, while one could say that part C is also contained in the robot, but,
the rule we wrote only returns the contained objects of the parts of the robot. To get part C, we would have to
add another rule that says that the contained objects of my contained objects are also contained in me, you can 
try that yourself and see if it works!


## To Cite:

```bib
@software{bassiouny2025rdr,
author = {Bassiouny, Abdelrhman},
title = {Ripple-Down-Rules},
url = {https://github.com/AbdelrhmanBassiouny/ripple_down_rules},
version = {0.5.2},
}
```
