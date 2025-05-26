# Ripple Down Rules


Welcome to the Ripple Down Rules package!
The ripple_down_rules is a python package that implements the various ripple down rules versions, including 
Single Classification (SCRDR), Multi Classification (MCRDR), and Generalised Ripple Down Rules (GRDR).

SCRDR, MCRDR, and GRDR are rule-based classifiers that are built incrementally, and can be used to classify data cases.
The rules are refined as new data cases are classified, this is done by prompting the user to add a new rule when a case
is misclassified or not classified at all. This allows the system to adapt to new data without the need for retraining.

SCRDR, MCRDR, and GRDR logic were inspired from the book: 
["Ripple Down Rules: An Alternative to Machine Learning"](https://www.taylorfrancis.com/books/mono/10.1201/9781003126157/ripple-rules-paul-compton-byeong-ho-kang) by Paul Compton, Byeong Ho Kang.


## 🚀 Enhanced Ripple-Down Rules Engine – Key Features

### 🧠 Ontology + Rule Base as One Entity
- 🧬 Unified data structure: Ontology and rules use the same Python data structures. 
- 🔄 Automatic sync: Updates to the ontology instantly reflect in the rule base. 
- 📦 Version controlled: The rule base is a Python module, versioned with your project.

### 🔁 Supports First, Second & Higher-Order Logic
- 🧩 Unlimited expressiveness: Rule conditions and conclusions are plain Python functions — anything Python can do, your rules can too!

### 🛡️ Automatic Rule Base Maintenance 
- ⚠️ Contradiction detection: New rules are auto-checked for logical conflicts. 
- 🔧 Prompted refinements: If a contradiction arises, you're guided to add a refinement rule. 

### 📝 Transparent & Editable Rule Base
- 📖 Readable rules: Rules are clean, understandable Python code. 
- 🔄 Reload-friendly: Easily edit and reload rules manually as needed.

### 💻 Developer-Centric Interface
- 👨‍💻 Feels like home: Seamless integration with your favorite IDE.
- ✨ Modern coding experience: Auto-completion and suggestions supported via IDE plugins.

### 🤖 LLM-Powered Rule Writing
- 💡 AI-assisted authoring: Ask AI for help or suggestions directly within the IDE.
- ⚡ Smart completion: Context-aware completions streamline rule writing.

### 🎯 Flexible Rule Specificity
- 🧪 Instance-level precision: Write rules for highly specific object scenarios.
- 🏛️ Generalization-ready: Create broad rules for superclass relationships.

### 🖼️ GUI for Rule Exploration
- 🧭 Object Explorer Panel: Navigate and inspect objects easily.
- 🧯 Interactive Diagram: Expandable/collapsible object diagram to guide rule creation visually.

This work aims to provide a flexible and powerful rule-based system that can be used in various applications,
from simple classification tasks to complex decision-making systems. Furthermore, one of the main goals is to
provide an easy-to-use interface that allows users to write rules in a natural way, without the need for
complex configurations or setups, and without the need to learn old or deprecated programming languages.

Future (and current) work will focus on improving the user experience, adding more features, and enhancing the
performance, so stay tuned for updates! and feel free to contribute, give feedback, or report issues on the
[GitHub repository](https://github.com/AbdelrhmanBassiouny/ripple_down_rules/issues)

## To Cite:

```bib
@software{bassiouny2025rdr,
author = {Bassiouny, Abdelrhman},
title = {Ripple-Down-Rules},
url = {https://github.com/AbdelrhmanBassiouny/ripple_down_rules},
version = {0.5.4},
}
```

## Installation
See [GitHub repository](https://github.com/AbdelrhmanBassiouny/ripple_down_rules)

## RippleDownRules (RDR)

This is the main abstract class for the ripple down rules. From this class, the different versions of
ripple down rules are derived. So the `SingleClassRDR`, `MultiClassRDR`, and `GeneralRDR` classes
are all derived from this class.

For most cases, you will use the `GeneralRDR` class, which is the most general version of the ripple down rules, 
and it internally uses the `SingleClassRDR` and `MultiClassRDR` classes to handle the classification tasks depending
on the case query.

This class has four main methods that you will mostly use:

- `fit`: This method is used to fit the rules to the data. It takes a list of `CaseQuery` objects that contain the
  data cases and their targets, and it will prompt the user to add rules when a case is misclassified or not classified at all.
- `fit_case`: This method is used to fit a single case to the rules.
- `classify`: This method is used to classify a data case. It takes a data case (any python object) and returns the
predicted target.
- `save`: This method is used to save the rules to a file. It will save the rules as a python module that can be imported
  in your project. In addition, it will save the rules as a JSON file with some metadata about the rules.
- `load`: This method is used to load the rules from a file. It will load the rules from a JSON file and then update the
  rules from the python module (which is important in case the user manually edited the rules in the python module).

### SingleClassRDR
This is the single classification version of the ripple down rules. It is used to classify data cases into a single
target class. This is used when your classification is mutually exclusive, meaning that a data case can only belong
to one class. For example, an animal can be either a "cat" or a "dog", but not both at the same time.

### MultiClassRDR
This is the multi classification version of the ripple down rules. It is used to classify data cases into multiple
target classes. This is used when your classification is not mutually exclusive, meaning that a data case can belong
to multiple classes at the same time. For example, an animal's habitat can be both "land" and "water" at the same time.

### GeneralRDR
This is the general version of the ripple down rules. It has the following features:
- It can handle both single and multi classification tasks by creating instances of `SingleClassRDR` and `MultiClassRDR`
  internally depending on the case query.
- It performs multiple passes over the rules and uses conclusions from previous passes to refine the classification,
    which allows it to handle complex classification tasks.

## Expert

The expert is an interface between the ripple down rules and the rule writer. Currently, only a Human expert is
implemented, but it is designed to be easily extendable to other types of experts, such as LLMs or other AI systems.

The main APIs are:

- `ask_for_conclusion`: This method is used to ask the expert for a conclusion or a target for a data case.
- `ask_for_conditions`: This method is used to ask the expert for the conditions that should be met for a rule to be
applied or evaluated.

## RDRDecorator

The `RDRDecorator` is a decorator that can be used to create rules in a more convenient way. It allows you to write 
functions normally and then decorate them with the `@RDRDecorator().decorator`. This will allow the function to be
to use ripple down rules to provide its output. This also allows you to write your own initial function logic, and then
this will be used input or feature to the ripple down rules.


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