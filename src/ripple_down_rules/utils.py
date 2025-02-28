from __future__ import annotations

import logging

import matplotlib
import networkx as nx
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from matplotlib import pyplot as plt
from ordered_set import OrderedSet
from sqlalchemy.orm import DeclarativeBase as Table, MappedColumn as Column
from tabulate import tabulate
from typing_extensions import Callable, Set, Any, Type, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ripple_down_rules.datastructures import Case, Attribute
    from ripple_down_rules.rules import Rule

matplotlib.use("Qt5Agg")  # or "Qt5Agg", depending on availability


def show_current_and_corner_cases(case: Case, targets: Optional[Union[List[Attribute], List[Column]]] = None,
                                  current_conclusions: Optional[Union[List[Attribute], List[Column]]] = None,
                                  last_evaluated_rule: Optional[Rule] = None) -> None:
    """
    Show the data of the new case and if last evaluated rule exists also show that of the corner case.

    :param case: The new case.
    :param targets: The target attribute of the case.
    :param current_conclusions: The current conclusions of the case.
    :param last_evaluated_rule: The last evaluated rule in the RDR.
    """
    corner_case = None
    if targets:
        targets = targets if isinstance(targets, list) else [targets]
    if current_conclusions:
        current_conclusions = current_conclusions if isinstance(current_conclusions, list) else [current_conclusions]
    targets = {f"target_{t.__class__.__name__}": t for t in targets} if targets else {}
    current_conclusions = {c.__class__.__name__: c for c in current_conclusions} if current_conclusions else {}
    if last_evaluated_rule:
        action = "Refinement" if last_evaluated_rule.fired else "Alternative"
        print(f"{action} needed for rule: {last_evaluated_rule}\n")
        corner_case = last_evaluated_rule.corner_case

    corner_row_dict = None
    if isinstance(case, Table):
        case_dict = row_to_dict(case)
        if last_evaluated_rule and last_evaluated_rule.fired:
            corner_row_dict = row_to_dict(last_evaluated_rule.corner_case)
    else:
        attributes = case._attributes_list
        if last_evaluated_rule and last_evaluated_rule.fired:
            attributes = OrderedSet(attributes + corner_case._attributes_list)
        names = [att._name for att in attributes]
        case_values = [case[name]._value if name in case._attributes else "null" for name in names]
        case_dict = dict(zip(names, case_values))
        if last_evaluated_rule and last_evaluated_rule.fired:
            corner_values = [corner_case[name]._value if name in corner_case._attributes else "null" for name in names]
            corner_row_dict = dict(zip(names, corner_values))

    if corner_row_dict:
        corner_conclusion = last_evaluated_rule.conclusion
        corner_row_dict.update({corner_conclusion.__class__.__name__: corner_conclusion})
        print_table_row(corner_row_dict)
    print("=" * 50)
    case_dict.update(targets)
    case_dict.update(current_conclusions)
    print_table_row(case_dict)


def print_table_row(row_dict: Dict[str, Any], columns_per_row: int = 9):
    """
    Print a table row.

    :param row_dict: The row to print.
    :param columns_per_row: The maximum number of columns per row.
    """
    all_items = list(row_dict.items())
    # make items a list of n rows such that each row has a max size of 4
    all_items = [all_items[i:i + columns_per_row] for i in range(0, len(all_items), columns_per_row)]
    keys = [list(map(lambda i: i[0], row)) for row in all_items]
    values = [list(map(lambda i: i[1], row)) for row in all_items]
    for row_keys, row_values in zip(keys, values):
        table = tabulate([row_values], headers=row_keys, tablefmt='plain')
        print(table)


def row_to_dict(obj):
    return {
        col.name: getattr(obj, col.name)
        for col in obj.__table__.columns
        if not col.primary_key and not col.foreign_keys
    }


def get_property_name(obj: Any, prop: Any) -> str:
    """
    Get the name of a property from an object.

    :param obj: The object to get the property name from.
    :param prop: The property to get the name of.
    """
    for name in dir(obj):
        if name.startswith("_") or callable(getattr(obj, name)):
            continue
        prop_value = getattr(obj, name)
        if prop_value is prop or (hasattr(prop_value, "_value") and prop_value._value is prop):
            return name


def get_attribute_values(obj: Any, attribute: Any) -> Any:
    """
    Get an attribute from a python object, if it is iterable, get the attribute values from all elements and unpack them
    into a list.

    :param obj: The object to get the sub attribute from.
    :param attribute: The  attribute to get.
    """
    if hasattr(obj, "__iter__") and not isinstance(obj, str):
        all_values = [get_attribute_values(a, attribute) for a in obj]
        if can_be_a_set(all_values):
            return set().union(*all_values)
        else:
            return set(all_values)
    return getattr(obj, attribute)


def can_be_a_set(value: Any) -> bool:
    """
    Check if a value can be a set.

    :param value: The value to check.
    """
    if hasattr(value, "__iter__") and not isinstance(value, str):
        if isinstance(value, set):
            return True
        if len(value) == 0:
            return True
        elif any(isinstance(v, (int, float, str, bool)) for v in value):
            return False
        else:
            return True
    else:
        return False


def get_all_subclasses(cls: Type) -> Dict[str, Type]:
    """
    Get all subclasses of a class recursively.

    :param cls: The class to get the subclasses of.
    :return: A dictionary of all subclasses.
    """
    all_subclasses: Dict[str, Type] = {}
    for sub_cls in cls.__subclasses__():
        all_subclasses[sub_cls.__name__.lower()] = sub_cls
        all_subclasses.update(get_all_subclasses(sub_cls))
    return all_subclasses


def make_set(value: Any) -> Set:
    """
    Make a set from a value.

    :param value: The value to make a set from.
    """
    if hasattr(value, "__iter__") and not isinstance(value, (str, type)):
        return set(value)
    return {value}


def make_value_or_raise_error(value: Any) -> Any:
    """
    Make a value or raise an error if the value is not a single value.

    :param value: The value to check.
    """
    if hasattr(value, "__iter__") and not isinstance(value, str):
        if hasattr(value, "__len__") and len(value) == 1:
            return list(value)[0]
        else:
            raise ValueError(f"Expected a single value, got {value}")
    return value


def tree_to_graph(root_node: Node) -> nx.DiGraph:
    """
    Convert anytree to a networkx graph.

    :param root_node: The root node of the tree.
    :return: A networkx graph.
    """
    graph = nx.DiGraph()
    unique_node_names = get_unique_node_names_func(root_node)

    def add_edges(node):
        if unique_node_names(node) not in graph.nodes:
            graph.add_node(unique_node_names(node))
        for child in node.children:
            if unique_node_names(child) not in graph.nodes:
                graph.add_node(unique_node_names(child))
            graph.add_edge(unique_node_names(node), unique_node_names(child), weight=child.weight)
            add_edges(child)

    add_edges(root_node)
    return graph


def get_unique_node_names_func(root_node) -> Callable[[Node], str]:
    nodes = [root_node]

    def get_all_nodes(node):
        for c in node.children:
            nodes.append(c)
            get_all_nodes(c)

    get_all_nodes(root_node)

    def nodenamefunc(node: Node):
        """
        Set the node name for the dot exporter.
        """
        similar_nodes = [n for n in nodes if n.name == node.name]
        node_idx = similar_nodes.index(node)
        return node.name if node_idx == 0 else f"{node.name}_{node_idx}"

    return nodenamefunc


def edge_attr_setter(parent, child):
    """
    Set the edge attributes for the dot exporter.
    """
    if child and hasattr(child, "weight") and child.weight:
        return f'style="bold", label=" {child.weight}"'
    return ""


def render_tree(root: Node, use_dot_exporter: bool = False,
                filename: str = "scrdr"):
    """
    Render the tree using the console and optionally export it to a dot file.

    :param root: The root node of the tree.
    :param use_dot_exporter: Whether to export the tree to a dot file.
    :param filename: The name of the file to export the tree to.
    """
    if not root:
        logging.warning("No rules to render")
        return
    for pre, _, node in RenderTree(root):
        print(f"{pre}{node.weight or ''} {node.__str__(sep='')}")
    if use_dot_exporter:
        unique_node_names = get_unique_node_names_func(root)

        de = DotExporter(root,
                         nodenamefunc=unique_node_names,
                         edgeattrfunc=edge_attr_setter
                         )
        de.to_dotfile(f"{filename}{'.dot'}")
        de.to_picture(f"{filename}{'.png'}")


def draw_tree(root: Node, fig: plt.Figure):
    """
    Draw the tree using matplotlib and networkx.
    """
    if root is None:
        return
    fig.clf()
    graph = tree_to_graph(root)
    fig_sz_x = 13
    fig_sz_y = 10
    fig.set_size_inches(fig_sz_x, fig_sz_y)
    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot")
    # scale down pos
    max_pos_x = max([v[0] for v in pos.values()])
    max_pos_y = max([v[1] for v in pos.values()])
    pos = {k: (v[0] * fig_sz_x / max_pos_x, v[1] * fig_sz_y / max_pos_y) for k, v in pos.items()}
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1000,
            ax=fig.gca(), node_shape="o", font_size=8)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'),
                                 ax=fig.gca(), rotate=False, clip_on=False)
    plt.pause(0.1)
