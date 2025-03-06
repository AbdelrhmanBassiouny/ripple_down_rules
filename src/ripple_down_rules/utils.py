from __future__ import annotations

import logging
from collections import UserDict

import matplotlib
import networkx as nx
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from matplotlib import pyplot as plt
from sqlalchemy import MetaData
from sqlalchemy.orm import Mapped, registry
from tabulate import tabulate
from typing_extensions import Callable, Set, Any, Type, Dict, TYPE_CHECKING, get_type_hints, \
    get_origin, get_args, Tuple, Optional, List

if TYPE_CHECKING:
    pass

matplotlib.use("Qt5Agg")  # or "Qt5Agg", depending on availability


def get_value_type_from_type_hint(attr_name: str, obj: Any) -> Type:
    """
    Get the value type from the type hint of an object attribute.

    :param attr_name: The name of the attribute.
    :param obj: The object to get the attributes from.
    """
    hint, origin, args = get_hint_for_attribute(attr_name, obj)
    if not origin:
        raise ValueError(f"Couldn't get type for Attribute {attr_name}, please provide a type hint")
    if origin in [list, set, tuple, type, dict]:
        attr_value_type = args[0]
    else:
        raise ValueError(f"Attribute {attr_name} has unsupported type {hint}.")
    return attr_value_type


def get_hint_for_attribute(attr_name: str, obj: Any) -> Tuple[Any, Any, Tuple[Any]]:
    """
    Get the type hint for an attribute of an object.

    :param attr_name: The name of the attribute.
    :param obj: The object to get the attribute from.
    :return: The type hint of the attribute.
    """
    class_attr = getattr(obj.__class__, attr_name)
    if isinstance(class_attr, property):
        if not class_attr.fget:
            raise ValueError(f"Attribute {attr_name} has no getter.")
        hint = get_type_hints(class_attr.fget)['return']
    else:
        hint = get_type_hints(obj.__class__)[attr_name]
    origin = get_origin(hint)
    args = get_args(hint)
    return hint, origin, args


def table_rows_as_str(row_dict: Dict[str, Any], columns_per_row: int = 9):
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
    all_table_rows = []
    for row_keys, row_values in zip(keys, values):
        table = tabulate([row_values], headers=row_keys, tablefmt='plain')
        all_table_rows.append(table)
    return "\n".join(all_table_rows)


def row_to_dict(obj):
    return {
        col.name: getattr(obj, col.name)
        for col in obj.__table__.columns
        if not col.primary_key and not col.foreign_keys
    }


def get_property_by_type(obj: Any, prop_type: Type) -> Optional[Any]:
    """
    Get a property from an object by type.

    :param obj: The object to get the property from.
    :param prop_type: The type of the property.
    """
    for name in dir(obj):
        if name.startswith("_") or callable(getattr(obj, name)):
            continue
        if isinstance(getattr(obj.__class__, name), (MetaData, registry)):
            continue
        prop_value = getattr(obj, name)
        if isinstance(prop_value, prop_type):
            return prop_value
        elif hasattr(prop_value, "__iter__") and not isinstance(prop_value, str):
            if len(prop_value) > 0 and any(isinstance(v, prop_type) for v in prop_value):
                return prop_value
            else:
                # get args of type hint
                hint, origin, args = get_hint_for_attribute(name, obj)
                if origin is Mapped:
                    origin, args = get_origin(args[0]), get_args(args[0])
                if origin in [list, set, tuple, dict, List, Set, Tuple, Dict]:
                    if prop_type is args[0]:
                        return prop_value
        else:
            # get the type hint of the attribute
            hint, origin, args = get_hint_for_attribute(name, obj)
            if origin is Mapped:
                if prop_type is args[0]:
                    return prop_value
            if origin is prop_type:
                return prop_value


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
        if prop_value is prop or (hasattr(prop_value, "_value") and prop_value.value is prop):
            return name


def get_attribute_values_transitively(obj: Any, attribute: Any) -> Any:
    """
    Get an attribute from a python object, if it is iterable, get the attribute values from all elements and unpack them
    into a list.

    :param obj: The object to get the sub attribute from.
    :param attribute: The  attribute to get.
    """
    if hasattr(obj, "__iter__") and not isinstance(obj, str):
        if isinstance(obj, (dict, UserDict)):
            all_values = [get_attribute_values_transitively(v, attribute) for v in obj.values()
                          if not isinstance(v, (str, type)) and hasattr(v, attribute)]
        else:
            all_values = [get_attribute_values_transitively(a, attribute) for a in obj
                          if not isinstance(a, (str, type)) and hasattr(a, attribute)]
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
        if len(value) > 0 and any(hasattr(v, "__iter__") and not isinstance(v, str) for v in value):
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
