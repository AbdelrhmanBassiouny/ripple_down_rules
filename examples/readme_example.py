from ripple_down_rules.rdr import SingleClassRDR, Category
from ripple_down_rules.datasets import load_zoo_dataset

all_cases, targets = load_zoo_dataset()

scrdr = SingleClassRDR()

# Fit the SCRDR to the data
scrdr.fit(all_cases, [Category(t) for t in targets],
          draw_tree=True, n_iter=10)

# Render the tree to a file
scrdr.render_tree(use_dot_exporter=True, filename="scrdr")

cat = scrdr.classify(all_cases[50])
assert cat.name == targets[50]
print(cat.name)
