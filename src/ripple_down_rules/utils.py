from __future__ import annotations

import ast
import importlib.util
import logging
import os
from _ast import AST

import matplotlib

from matplotlib import pyplot as plt

import networkx as nx
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from ordered_set import OrderedSet
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from sqlalchemy.orm import DeclarativeBase as Table, Session, MappedColumn as Column
from tabulate import tabulate
from typing_extensions import Callable, Set, Any, Type, Dict, List, Tuple, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ripple_down_rules.datastructures import Case, PromptFor, Attribute
    from ripple_down_rules.rules import Rule

matplotlib.use("Qt5Agg")  # or "Qt5Agg", depending on availability


class CallableExpression:
    """
    A callable that is constructed from a string statement written by an expert.
    """
    conclusion_type: Type
    """
    The type of the output of the callable, used for assertion.
    """

    def __init__(self, user_input: str, conclusion_type: Type, expression_tree: Optional[AST] = None,
                 session: Optional[Session] = None):
        """
        Create a callable expression.

        :param user_input: The input given by the expert.
        :param conclusion_type: The type of the output of the callable.
        :param expression_tree: The AST tree parsed from the user input.
        :param session: The sqlalchemy orm session.
        """
        self.user_input: str = user_input
        if not expression_tree:
            expression_tree = parse_string_to_expression(user_input)
        self.expression_tree: AST = expression_tree
        self.conclusion_type = conclusion_type
        self.session = session
        self.visitor = VariableVisitor()
        self.visitor.visit(expression_tree)
        self.code = compile_expression_to_code(expression_tree)

    def __call__(self, case: Any, **kwargs) -> conclusion_type:
        try:
            context = get_all_possible_contexts(case)
            assert_context_contains_needed_information(case, context, self.visitor)
            output = eval(self.code, {"__builtins__": {"len": len}}, context)
            if self.session:
                output = self.add_row_and_query_expression_result(case, output)
            assert isinstance(output, self.conclusion_type), (f"Expected output type {self.conclusion_type},"
                                                              f" got {type(output)}")
            return output
        except Exception as e:
            raise ValueError(f"Error during evaluation: {e}")

    def add_row_and_query_expression_result(self, case: Table, evaluated_expression: Any) -> Callable[Table, Any]:
        """
        Evaluate a sqlalchemy statement written by an expert, this is done by inserting the case in parent table and
        querying the data needed for the expert statement from the table using the sqlalchemy orm session.

        :param case: The case about which is input is given.
        :param evaluated_expression: The statement given by the expert.
        """
        table = case.__class__
        self.session.add(case)
        self.session.commit()
        return self.session.query(table).filter(table.id == case.id, evaluated_expression).first()

    def __str__(self):
        return self.user_input


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
    targets = {f"target_{t.__class__.__name__}": t for t in targets} if targets else {}
    current_conclusions = {c.__class__.__name__: c for c in current_conclusions} if current_conclusions else {}
    if last_evaluated_rule:
        action = "Refinement" if last_evaluated_rule.fired else "Alternative"
        print(f"{action} needed for rule:\n")
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
        corner_row_dict.update(targets)
        corner_row_dict.update(current_conclusions)
        print_table_row(corner_row_dict)

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
        table = tabulate([row_values], headers=row_keys, tablefmt='grid')
        print(table)


def row_to_dict(obj):
    return {
        col.name: getattr(obj, col.name)
        for col in obj.__table__.columns
        if not col.primary_key and not col.foreign_keys
    }


def prompt_user_for_expression(case: Union[Case, Table], prompt_for: PromptFor, target_name: str,
                               output_type: Type, session: Optional[Session] = None) -> Tuple[str, CallableExpression]:
    """
    Prompt the user for an executable python expression.

    :param case: The case to classify.
    :param prompt_for: The type of information ask user about.
    :param target_name: The name of the target attribute to compare the case with.
    :param output_type: The type of the output of the given statement from the user.
    :param session: The sqlalchemy orm session.
    :return: A callable expression that takes a case and executes user expression on it.
    """
    user_input, expression_tree = prompt_user_about_case(case, prompt_for, target_name)
    callable_expression = CallableExpression(user_input, output_type, expression_tree=expression_tree, session=session)
    return user_input, callable_expression


def prompt_user_about_case(case: Union[Case, Table], prompt_for: PromptFor, target_name: str) \
        -> Tuple[str, str]:
    """
    Prompt the user for input.

    :param case: The case to prompt the user on.
    :param prompt_for: The type of information the user should provide for the given case.
    :param target_name: The name of the target property of the case that is queried.
    :return: The user input, and the executable expression that was parsed from the user input.
    """
    prompt_str = f"Give {prompt_for} for {case.__class__.__name__}.{target_name}"
    session = get_prompt_session_for_obj(case)
    user_input, expression_tree = prompt_user_input_and_parse_to_expression(prompt_str, session)
    return user_input, expression_tree


def evaluate_alchemy_expression(case: Table, session: Session, expert_input: str,
                                expression_tree: Optional[AST] = None) -> Callable[Table, Any]:
    """
    Evaluate a sqlalchemy statement written by an expert, this is done by inserting the case in parent table and
    querying the data needed for the expert statement from the table using the sqlalchemy orm session.

    :param case: The case about which is input is given.
    :param session: The sqlalchemy orm session.
    :param expert_input: The statement given by the expert.
    :param expression_tree: The AST tree parsed from the expert input, if not give it will be parsed from user_input.
    """
    if not expression_tree:
        expression_tree = parse_string_to_expression(expert_input)
    code = compile_expression_to_code(expression_tree)
    condition: Any = eval(code, {"__builtins__": {"len": len}})
    table = case.__class__
    session.add(case)
    session.commit()
    return lambda new_case: session.query(table).filter(table.id == new_case.id, condition).first()


def compile_expression_to_code(expression_tree: AST) -> Any:
    """
    Compile an expression tree that was parsed from string into code that can be executed using 'eval(code)'

    :param expression_tree: The parsed expression tree.
    :return: The code that was compiled from the expression tree.
    """
    return compile(expression_tree, filename="<string>", mode="eval")


def assert_context_contains_needed_information(case: Union[Case, Table], context: Dict[str, Any],
                                               visitor: VariableVisitor):
    """
    Asserts that the variables mentioned in the expression visited by visitor are all in the given context.
    """
    for key in visitor.variables:
        if key not in context:
            raise ValueError(f"Attribute {key} not found in the case {case}")
    for key, ast_attr in visitor.attributes.items():
        if f"{key.id}.{ast_attr.attr}" not in context:
            raise ValueError(f"Attribute {key.id}.{ast_attr.attr} not found in the case {case}")


def get_all_possible_contexts(obj: Any, recursion_idx: int = 0, max_recursion_idx: int = 2) -> Dict[str, Any]:
    """
    Get all possible contexts for an object.

    :param obj: The object to get the contexts for.
    :param recursion_idx: The recursion index to prevent infinite recursion.
    :param max_recursion_idx: The maximum recursion index.
    :return: A dictionary of all possible contexts.
    """
    all_contexts = {}
    if recursion_idx > max_recursion_idx:
        return all_contexts
    for attr in dir(obj):
        if attr.startswith("__") or attr.startswith("_") or callable(getattr(obj, attr)):
            continue
        all_contexts[attr] = get_attribute_values(obj, attr)
        sub_attr_contexts = get_all_possible_contexts(getattr(obj, attr), recursion_idx + 1)
        sub_attr_contexts = {f"{attr}.{k}": v for k, v in sub_attr_contexts.items()}
        all_contexts.update(sub_attr_contexts)
    return all_contexts


def prompt_user_input_and_parse_to_expression(prompt: Optional[str] = None, session: Optional[PromptSession] = None,
                                              user_input: Optional[str] = None) -> Tuple[str, ast.AST]:
    """
    Prompt the user for input.

    :param prompt: The prompt to display to the user.
    :param session: The prompt session to use.
    :param user_input: The user input to use. If given, the user input will be used instead of prompting the user.
    :return: The user input and the AST tree.
    """
    while True:
        if not user_input:
            user_input = session.prompt(f"\n{prompt} >>> ")
        if user_input.lower() in ['exit', 'quit', '']:
            break
        try:
            return user_input, parse_string_to_expression(user_input)
        except SyntaxError as e:
            print(f"Syntax error: {e}")


def parse_string_to_expression(expression_str: str) -> AST:
    """
    Parse a string statement into an AST expression.

    :param expression_str: The string which will be parsed.
    :return: The parsed expression.
    """
    tree = ast.parse(expression_str, mode='eval')
    logging.debug(f"AST parsed successfully: {ast.dump(tree)}")
    return tree


def get_prompt_session_for_obj(obj: Any) -> PromptSession:
    """
    Get a prompt session for an object.

    :param obj: The object to get the prompt session for.
    :return: The prompt session.
    """
    completions = get_completions(obj)
    completer = WordCompleter(completions)
    session = PromptSession(completer=completer)
    return session


class VariableVisitor(ast.NodeVisitor):
    """
    A visitor to extract all variables and comparisons from a python expression represented as an AST tree.
    """
    compares: List[Tuple[Union[ast.Name, ast.Call], ast.cmpop, Union[ast.Name, ast.Call]]]
    variables: Set[str]
    all: List[ast.BoolOp]

    def __init__(self):
        self.variables = set()
        self.attributes: Dict[ast.Name, ast.Attribute] = {}
        self.compares = list()
        self.all = list()

    def visit_Attribute(self, node):
        self.all.append(node)
        self.attributes[node.value] = node
        self.generic_visit(node)

    def visit_BinOp(self, node):
        self.all.append(node)
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.all.append(node)
        self.generic_visit(node)

    def visit_Compare(self, node):
        self.all.append(node)
        self.compares.append([node.left, node.ops[0], node.comparators[0]])
        self.generic_visit(node)

    def visit_Name(self, node):
        if f"__{node.id}__" not in dir(__builtins__) and node not in self.attributes:
            self.variables.add(node.id)
        self.generic_visit(node)


def get_property_name(obj: Any, prop: Any) -> str:
    """
    Get the name of a property from an object.

    :param obj: The object to get the property name from.
    :param prop: The property to get the name of.
    """
    for name in dir(obj):
        if name.startswith("_"):
            continue
        prop_value = getattr(obj, name)
        if prop_value is prop or (hasattr(prop_value, "_value") and prop_value._value is prop):
            return name


def get_completions(obj: Any) -> List[str]:
    """
    Get all completions for the object. This is used in the python prompt shell to provide completions for the user.

    :param obj: The object to get completions for.
    :return: A list of completions.
    """
    # Define completer with all object attributes and comparison operators
    completions = ['==', '!=', '>', '<', '>=', '<=', 'in', 'not', 'and', 'or', 'is']
    completions += ["isinstance(", "issubclass(", "type(", "len(", "hasattr(", "getattr(", "setattr(", "delattr("]
    completions += list(get_all_possible_contexts(obj).keys())
    return completions


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


def import_class(class_name: str, file_path: str = "dynamically_created_attributes.py"):
    module_name = os.path.splitext(file_path)[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Access the class from the module
    new_class = getattr(module, class_name)
    print(f"Class {class_name} imported dynamically.")
    return new_class


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
