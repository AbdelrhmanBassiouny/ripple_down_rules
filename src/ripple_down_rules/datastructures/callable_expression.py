from __future__ import annotations

import ast
import logging
from _ast import AST

from sqlalchemy.orm import Session
from typing_extensions import Type, Optional, Any, List, Union, Tuple, Dict, Set

from .case import create_case, Case
from ..utils import SubclassJSONSerializer, get_full_class_name, get_type_from_string


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
        self.types = set()
        self.callables = set()
        self.compares = list()
        self.binary_ops = list()
        self.all = list()

    def visit_Constant(self, node):
        self.all.append(node)
        self.types.add(node)
        self.generic_visit(node)

    def visit_Call(self, node):
        self.all.append(node)
        self.callables.add(node)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.all.append(node)
        self.attributes[node.value] = node
        self.generic_visit(node)

    def visit_BinOp(self, node):
        self.binary_ops.append(node)
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


def get_used_scope(code_str, scope):
    # Parse the code into an AST
    mode = 'exec' if code_str.startswith('def') else 'eval'
    tree = ast.parse(code_str, mode=mode)

    # Walk the AST to collect used variable names
    class NameCollector(ast.NodeVisitor):
        def __init__(self):
            self.names = set()

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):  # We care only about variables being read
                self.names.add(node.id)

    collector = NameCollector()
    collector.visit(tree)

    # Filter the scope to include only used names
    used_scope = {k: scope[k] for k in collector.names if k in scope}
    return used_scope


class CallableExpression(SubclassJSONSerializer):
    """
    A callable that is constructed from a string statement written by an expert.
    """
    conclusion_type: Type
    """
    The type of the output of the callable, used for assertion.
    """
    expression_tree: AST
    """
    The AST tree parsed from the user input.
    """
    user_input: str
    """
    The input given by the expert.
    """
    session: Optional[Session]
    """
    The sqlalchemy orm session.
    """
    visitor: VariableVisitor
    """
    A visitor to extract all variables and comparisons from a python expression represented as an AST tree.
    """
    code: Any
    """
    The code that was compiled from the expression tree
    """
    compares_column_offset: List[int]
    """
    The start and end indices of each comparison in the string of user input.
    """

    def __init__(self, user_input: str, conclusion_type: Optional[Type] = None, expression_tree: Optional[AST] = None,
                 scope: Optional[Dict[str, Any]] = None):
        """
        Create a callable expression.

        :param user_input: The input given by the expert.
        :param conclusion_type: The type of the output of the callable.
        :param expression_tree: The AST tree parsed from the user input.
        :param session: The sqlalchemy orm session.
        """
        self.user_input: str = user_input
        self.conclusion_type = conclusion_type
        self.scope: Optional[Dict[str, Any]] = scope if scope is not None else {}
        self.scope = get_used_scope(self.user_input, self.scope)
        self.update_expression(expression_tree)

    def get_used_scope_in_user_input(self) -> Set[str]:
        """
        Get the used scope in the user input.
        :return: The used scope in the user input.
        """
        return self.visitor.variables.union(self.visitor.attributes.keys())

    def update_expression(self, expression_tree: Optional[AST] = None):
        if not expression_tree:
            expression_tree = parse_string_to_expression(self.user_input)
        self.expression_tree: AST = expression_tree
        self.visitor = VariableVisitor()
        self.visitor.visit(expression_tree)
        self.expression_tree = parse_string_to_expression(self.user_input)
        self.compares_column_offset = [(c[0].col_offset, c[2].end_col_offset) for c in self.visitor.compares]
        self.code = compile_expression_to_code(self.expression_tree)

    def __call__(self, case: Any, **kwargs) -> Any:
        try:
            if not isinstance(case, Case):
                case = create_case(case, max_recursion_idx=3)
            scope = {'case': case, **self.scope}
            output = eval(self.code, scope)
            if output is None:
                output = scope['_get_value'](case)
            if self.conclusion_type is not None:
                assert isinstance(output, self.conclusion_type), (f"Expected output type {self.conclusion_type},"
                                                                  f" got {type(output)}")
            return output
        except Exception as e:
            raise ValueError(f"Error during evaluation: {e}")

    def combine_with(self, other: 'CallableExpression') -> 'CallableExpression':
        """
        Combine this callable expression with another callable expression using the 'and' operator.
        """
        new_user_input = f"({self.user_input}) and ({other.user_input})"
        return CallableExpression(new_user_input, conclusion_type=self.conclusion_type)

    def __str__(self):
        """
        Return the user string where each compare is written in a line using compare column offset start and end.
        """
        binary_ops = sorted(self.visitor.binary_ops, key=lambda x: x.end_col_offset)
        binary_ops_indices = [b.end_col_offset for b in binary_ops]
        all_binary_ops = []
        prev_e = 0
        for i, e in enumerate(binary_ops_indices):
            if i == 0:
                all_binary_ops.append(self.user_input[:e])
            else:
                all_binary_ops.append(self.user_input[prev_e:e])
            prev_e = e
        return "\n".join(all_binary_ops) if len(all_binary_ops) > 0 else self.user_input

    def _to_json(self) -> Dict[str, Any]:
        return {"user_input": self.user_input, "conclusion_type": get_full_class_name(self.conclusion_type),
                "scope": {k: get_full_class_name(v) for k, v in self.scope.items()
                          if hasattr(v, '__module__') and hasattr(v, '__name__')}
                }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> CallableExpression:
        return cls(user_input=data["user_input"], conclusion_type=get_type_from_string(data["conclusion_type"]),
                   scope={k: get_type_from_string(v) for k, v in data["scope"].items()})


def compile_expression_to_code(expression_tree: AST) -> Any:
    """
    Compile an expression tree that was parsed from string into code that can be executed using 'eval(code)'

    :param expression_tree: The parsed expression tree.
    :return: The code that was compiled from the expression tree.
    """
    mode = 'exec' if isinstance(expression_tree, ast.Module) else 'eval'
    return compile(expression_tree, filename="<string>", mode=mode)


def assert_context_contains_needed_information(case: Any, context: Dict[str, Any],
                                               visitor: VariableVisitor) -> Tuple[Set[str], Set[str]]:
    """
    Asserts that the variables mentioned in the expression visited by visitor are all in the given context.

    :param case: The case to check the context for.
    :param context: The context to check.
    :param visitor: The visitor that visited the expression.
    :return: The found variables and attributes.
    """
    found_variables = set()
    for key in visitor.variables:
        if key not in context:
            raise ValueError(f"Variable {key} not found in the case {case}")
        found_variables.add(key)

    found_attributes = get_attributes_str(visitor)
    for attr in found_attributes:
        if attr not in context:
            raise ValueError(f"Attribute {attr} not found in the case {case}")
    return found_variables, found_attributes


def get_attributes_str(visitor: VariableVisitor) -> Set[str]:
    """
    Get the string representation of the attributes in the given visitor.

    :param visitor: The visitor that visited the expression.
    :return: The string representation of the attributes.
    """
    found_attributes = set()
    for key, ast_attr in visitor.attributes.items():
        str_attr = ""
        while isinstance(key, ast.Attribute):
            if len(str_attr) > 0:
                str_attr = f"{key.attr}.{str_attr}"
            else:
                str_attr = key.attr
            key = key.value
        str_attr = f"{key.id}.{str_attr}" if len(str_attr) > 0 else f"{key.id}.{ast_attr.attr}"
        found_attributes.add(str_attr)
    return found_attributes


def parse_string_to_expression(expression_str: str) -> AST:
    """
    Parse a string statement into an AST expression.

    :param expression_str: The string which will be parsed.
    :return: The parsed expression.
    """
    mode = 'exec' if expression_str.startswith('def') else 'eval'
    tree = ast.parse(expression_str, mode=mode)
    logging.debug(f"AST parsed successfully: {ast.dump(tree)}")
    return tree
