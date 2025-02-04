import networkx as nx
from anytree import Node


def tree_to_graph(root_node: Node) -> nx.DiGraph:
    """
    Convert anytree to a networkx graph.

    :param root_node: The root node of the tree.
    :return: A networkx graph.
    """
    graph = nx.DiGraph()

    def add_edges(node):
        if node not in graph.nodes:
            graph.add_node(node.name)
        for child in node.children:
            if node not in graph.nodes:
                graph.add_node(child.name)
            graph.add_edge(node.name, child.name, weight=child.weight)
            add_edges(child)

    add_edges(root_node)
    return graph
