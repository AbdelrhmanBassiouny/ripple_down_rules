from __future__ import annotations

import ast
import importlib.util
import logging
import os
from _ast import AST

import networkx as nx
from sqlalchemy.orm import DeclarativeBase, Session
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from matplotlib import pyplot as plt
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from sqlalchemy import BinaryExpression, Engine
from tabulate import tabulate
from typing_extensions import Callable, Set, Any, Type, Dict, List, Tuple, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ripple_down_rules.datastructures import Case, ObjectAttributeTarget, Condition, PromptFor, ExpressionParser

Table: Type[DeclarativeBase]


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


def prompt_for_alchemy_conditions(x: Table, target: ObjectAttributeTarget, session: Session,
                                  user_input: Optional[str] = None) -> Tuple[str, BinaryExpression]:
    """
    Prompt the user for relational conditions.

    :param x: The case to classify.
    :param target: The target category to compare the case with.
    :param session: The SQLAlchemy ORM session to use.
    :param user_input: The user input to parse. If None, the user is prompted for input.
    :return: The differentiating features as new rule conditions.
    """
    session = get_prompt_session_for_obj(x)
    prompt_str = f"Give Conditions for {x.__tablename__}.{target.name}"
    user_input, tree = prompt_user_input_and_parse_to_expression(prompt_str, session, user_input=user_input)
    condition = evaluate_alchemy_condition(x, user_input, tree, bool)
    return user_input, condition


def prompt_for_relational_conditions(case: Union[Case, Table], target: ObjectAttributeTarget,
                                     user_input: Optional[str] = None) -> Tuple[str, Condition]:
    """
    Prompt the user for relational conditions.

    :param case: The case to classify.
    :param target: The target category to compare the case with.
    :param user_input: The user input to parse. If None, the user is prompted for input.
    :return: The differentiating features as new rule conditions.
    """
    user_input, expression_tree = prompt_user_about_case(case, PromptFor.Conditions, target)
    conditions = parse_expression_to_callable(case, expression_tree, user_input=user_input, callable_output_type=bool)
    return user_input, conditions


def parse_expression_to_callable(case: Union[Case, Table], expression_tree: AST,
                                 user_input: Optional[str] = None, callable_output_type: Optional[Type] = None,
                                 expression_parser: ExpressionParser = ExpressionParser.ASTVisitor,
                                 session: Optional[Session] = None):
    if expression_parser == ExpressionParser.ASTVisitor:
        conditions = parse_relational_input(case, user_input, expression_tree, callable_output_type)
    elif expression_parser == ExpressionParser.Alchemy:
        conditions = evaluate_alchemy_expression(case, session, expression_tree)
    else:
        raise ValueError(f"Incorrect case type {type(case)}, case should be either a Case or an ORM Table")
    return conditions


def prompt_user_about_case(case: Union[Case, Table], prompt_for: PromptFor, target: ObjectAttributeTarget)\
        -> Tuple[str, str]:
    """
    Prompt the user for input.

    :param case: The case to prompt the user on.
    :param prompt_for: The type of information the user should provide for the given case.
    :return: The user input, and the executable expression that was parsed from the user input.
    """
    prompt_str = f"Give {prompt_for} for {case.__class__.__name__}.{target.name}"
    session = get_prompt_session_for_obj(case)
    user_input, expression_tree = prompt_user_input_and_parse_to_expression(prompt_str, session)
    return user_input, expression_tree


def evaluate_alchemy_expression(case: Table, session: Session, expression_tree: AST) -> Any:
    code = compile_expression_to_code(expression_tree)
    condition: BinaryExpression = eval(code, {"__builtins__": {"len": len}})
    table = case.__class__
    session.add(case)
    session.commit()
    result = session.query(table).filter(table.id == case.id, condition).first()


def compile_expression_to_code(expression_tree: AST) -> Any:
    """
    Compile an expression tree that was parsed from string into code that can be executed using 'eval(code)'

    :param expression_tree: The parsed expression tree.
    :return: The code that was compiled from the expression tree.
    """
    return compile(expression_tree, filename="<string>", mode="eval")


def parse_relational_input(corner_case: Case, user_input: str, expression_tree: ast.AST, conclusion_type: Type) -> Callable[[Case], bool]:
    """
    Parse the relational information from the user input.

    :param corner_case: The case to classify.
    :param user_input: The input to parse.
    :param expression_tree: The AST tree of the input.
    :param conclusion_type: The output type of the evaluation of the parsed input.
    :return: The parsed conditions as a dictionary.
    """
    visitor = VariableVisitor()
    visitor.visit(expression_tree)

    code = compile_expression_to_code(expression_tree)

    class CallableExpression:

        def __call__(self, case: Any, **kwargs) -> conclusion_type:
            try:
                context = get_all_possible_contexts(case)
                assert_context_contains_needed_information(case, context, visitor)
                output = eval(code, {"__builtins__": {"len": len}}, context)
                assert isinstance(output, conclusion_type), (f"Expected output type {conclusion_type},"
                                                             f" got {type(output)}")
                return output
            except Exception as e:
                raise ValueError(f"Error during evaluation: {e}")

        def __str__(self):
            return user_input

    return CallableExpression()


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


def get_all_possible_contexts(obj: Any, recursion_idx: int = 0) -> Dict[str, Any]:
    """
    Get all possible contexts for an object.

    :param obj: The object to get the contexts for.
    :return: A dictionary of all possible contexts.
    """
    all_contexts = {}
    if recursion_idx > 2:
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
            # Parse the input into an AST
            tree = ast.parse(user_input, mode='eval')
            logging.debug(f"AST parsed successfully: {ast.dump(tree)}")
            return user_input, tree
        except SyntaxError as e:
            print(f"Syntax error: {e}")


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
