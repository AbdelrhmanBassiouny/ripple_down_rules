from ripple_down_rules.rdr import SingleClassRDR
from ripple_down_rules.datasets import load_zoo_dataset
from ripple_down_rules.utils import render_tree

all_cases, targets = load_zoo_dataset()

scrdr = SingleClassRDR()

# Fit the SCRDR to the data
scrdr.fit(all_cases, targets,
          animate_tree=True, n_iter=10)

# Render the tree to a file
render_tree(scrdr.start_rule, use_dot_exporter=True, filename="scrdr")

cat = scrdr.fit_case(all_cases[50], targets[50])
assert cat == targets[50]
