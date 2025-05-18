import graphviz

from ripple_down_rules.datasets import load_zoo_dataset, Species
from ripple_down_rules.datastructures.dataclasses import CaseQuery


def is_simple(obj):
    return isinstance(obj, (int, float, str, bool, type(None)))

def get_colored_value(value):
    if isinstance(value, str):
        color = '#A31515'  # red for strings
        val = repr(value)
    elif isinstance(value, (int, float)):
        color = '#098658'  # green for numbers
        val = str(value)
    elif isinstance(value, bool):
        color = '#0000FF'  # blue for booleans
        val = str(value)
    elif value is None:
        color = '#808080'  # gray for None
        val = 'None'
    else:
        color = '#000000'  # fallback
        val = str(value)
    return f'<FONT COLOR="{color}">{val}</FONT>'

def generate_object_graph(obj, name='root', seen=None, graph=None):
    if seen is None:
        seen = set()
    if graph is None:
        graph = graphviz.Digraph(format='png')
        graph.attr('node', shape='plaintext')

    obj_id = id(obj)
    if obj_id in seen:
        return graph
    seen.add(obj_id)

    # Build HTML table label for this object node
    rows = [f'<TR><TD><B>{name}</B></TD><TD><I>{type(obj).__name__}</I></TD></TR>']

    # We'll keep track of non-simple attrs to add edges later
    non_simple_attrs = []

    if hasattr(obj, '__dict__'):
        for attr, value in vars(obj).items():
            if attr.startswith('_'):
                continue
            if attr == 'scope':
                continue
            if is_simple(value):
                val_colored = get_colored_value(value)
                rows.append(f'<TR><TD ALIGN="LEFT" PORT="{attr}">{attr}</TD><TD ALIGN="LEFT">{val_colored}</TD></TR>')
            else:
                # Show just name and type inside the node
                rows.append(f'<TR><TD ALIGN="LEFT" PORT="{attr}">{attr}</TD><TD ALIGN="LEFT"><I>{type(value).__name__}</I></TD></TR>')
                non_simple_attrs.append((attr, value))

    elif isinstance(obj, (list, tuple, set, dict)):
        items = obj.items() if isinstance(obj, dict) else enumerate(obj)
        for idx, item in items:
            if idx == "scope":
                continue
            # Represent items as attr = index + type (for the label)
            if is_simple(item):
                val_colored = get_colored_value(item)
                rows.append(f'<TR><TD ALIGN="LEFT" PORT="{idx}">[{idx}]</TD><TD ALIGN="LEFT">{val_colored}</TD></TR>')
            else:
                rows.append(f'<TR><TD ALIGN="LEFT" PORT="{idx}">[{idx}]</TD><TD ALIGN="LEFT"><I>{type(item).__name__}</I></TD></TR>')
                non_simple_attrs.append((str(idx), item))

    label = f"""<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
        {''.join(rows)}
        </TABLE>
    >"""

    graph.node(str(obj_id), label)

    # Add edges from this node to non-simple attribute nodes
    for attr, value in non_simple_attrs:
        generate_object_graph(value, attr, seen, graph)
        # Edge from this object's attribute port to nested object's node
        graph.edge(f"{obj_id}:{attr}", str(id(value)), label=attr)

    return graph



class Address:
    def __init__(self, city):
        self.city = city

class Person:
    def __init__(self, name, address):
        self.name = name
        self.address = address


if __name__ == "__main__":
    # Example usage
    home = Address("Cairo")
    p = Person("Ahmed", home)

    cases, targets = load_zoo_dataset(cache_file="zoo")
    cq = CaseQuery(cases[0], "species", (Species,), True, _target=targets[0])

    graph = generate_object_graph(cq)
    graph.render('object_diagram', view=True)
