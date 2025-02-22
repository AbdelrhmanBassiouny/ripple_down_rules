import logging

import networkx as nx
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from matplotlib import pyplot as plt
from typing_extensions import Callable, Set, Any


def make_set(value: Any) -> Set:
    """
    Make a set from a value.

    :param value: The value to make a set from.
    """
    if hasattr(value, "__iter__") and not isinstance(value, str):
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
