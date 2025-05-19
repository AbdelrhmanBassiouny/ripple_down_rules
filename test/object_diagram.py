from enum import Enum

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
    val = (val.replace("class ", "").replace("'", "").replace('<', "")
           .replace('>', "").replace('?', '').replace('\n', ' '))
    val = val[:50] + '...' if len(val) > 50 else val
    return f'<FONT COLOR="{color}">{val}</FONT>'

def generate_object_graph(obj, name='root', seen=None, graph=None, current_depth=0, max_depth=3):
    if seen is None:
        seen = set()
    if graph is None:
        graph = graphviz.Digraph(format='png')
        graph.attr('node', shape='plaintext')

    obj_id = id(obj)
    if obj_id in seen or current_depth > max_depth:
        return graph
    seen.add(obj_id)

    # Build HTML table label for this object node
    type_name = type(obj).__name__ if not isinstance(obj, type) else obj.__name__
    rows = [f'<TR><TD><B>{name}</B></TD><TD><I>{type_name}</I></TD></TR>']

    # We'll keep track of non-simple attrs to add edges later
    non_simple_attrs = []

    if isinstance(obj, (list, tuple, set, dict)):
        items = obj.items() if isinstance(obj, dict) else enumerate(obj)
        for idx, item in items:
            if idx == "scope":
                continue
            # Represent items as attr = index + type (for the label)
            if is_simple(item):
                val_colored = get_colored_value(item)
                rows.append(f'<TR><TD ALIGN="LEFT" PORT="{idx}">[{idx}]</TD><TD ALIGN="LEFT">{val_colored}</TD></TR>')
            else:
                type_name = type(item).__name__ if not isinstance(item, type) else item.__name__
                rows.append(f'<TR><TD ALIGN="LEFT" PORT="{idx}">[{idx}]</TD><TD ALIGN="LEFT"><I>{type_name}</I></TD></TR>')
                non_simple_attrs.append((str(idx), item))

    else:
        for attr in dir(obj):
            if attr.startswith('_'):
                continue
            if attr == 'scope':
                continue
            value = getattr(obj, attr)
            if callable(value):
                continue
            if is_simple(value):
                val_colored = get_colored_value(value)
                rows.append(f'<TR><TD ALIGN="LEFT" PORT="{attr}">{attr}</TD><TD ALIGN="LEFT">{val_colored}</TD></TR>')
            else:
                type_name = type(value).__name__ if not isinstance(value, type) else value.__name__
                # Show just name and type inside the node
                rows.append(f'<TR><TD ALIGN="LEFT" PORT="{attr}">{attr}</TD><TD ALIGN="LEFT"><I>{type_name}</I></TD></TR>')
                non_simple_attrs.append((attr, value))

    label = f"""<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
        {''.join(rows)}
        </TABLE>
    >"""

    graph.node(str(obj_id), label)

    # Add edges from this node to non-simple attribute nodes
    i = 0
    for attr, value in non_simple_attrs:
        if attr.startswith('_'):
            continue
        if attr == 'scope':
            continue
        if callable(value):
            continue
        generate_object_graph(value, attr, seen, graph, current_depth + 1, max_depth)
        # Edge from this object's attribute port to nested object's node
        graph.edge(f"{obj_id}:{attr}", str(id(value)), label=attr)
        i += 1

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
