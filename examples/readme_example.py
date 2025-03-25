from ripple_down_rules.datastructures import CaseQuery
from ripple_down_rules.rdr import SingleClassRDR
from ripple_down_rules.datasets import load_zoo_dataset
from ripple_down_rules.utils import render_tree

all_cases, targets = load_zoo_dataset()

scrdr = SingleClassRDR()

# Fit the SCRDR to the data
case_queries = [CaseQuery(case, target=target) for case, target in zip(all_cases, targets)]
scrdr.fit(case_queries, animate_tree=True)

# Render the tree to a file
render_tree(scrdr.start_rule, use_dot_exporter=True, filename="scrdr")

cat = scrdr.fit_case(all_cases[50], targets[50])
assert cat == targets[50]
