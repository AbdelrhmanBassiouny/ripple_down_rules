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

```bash

## Example Usage

Fit the SCRDR to the data, then classify one of the data cases to check if its correct,
and render the tree to a file:
```

```python
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.rdr import SingleClassRDR
from ripple_down_rules.datasets import load_zoo_dataset
from ripple_down_rules.utils import render_tree

all_cases, targets = load_zoo_dataset()

scrdr = SingleClassRDR()

# Fit the SCRDR to the data
case_queries = [CaseQuery(case, 'species', type(target), True, _target=target)
                for case, target in zip(all_cases[:10], targets[:10])]
scrdr.fit(case_queries, animate_tree=True)

# Render the tree to a file
render_tree(scrdr.start_rule, use_dot_exporter=True, filename="scrdr")

cat = scrdr.classify(all_cases[50])
assert cat == targets[50]
```
```bib
@software{bassiouny2025rdr,
author = {Bassiouny, Abdelrhman},
title = {Ripple-Down-Rules},
url = {https://github.com/AbdelrhmanBassiouny/ripple_down_rules},
version = {0.4.1},
}
```
