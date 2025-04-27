from __future__ import annotations

import ast
import logging
from _ast import AST

from typing_extensions import Type, Optional, Any, List, Union, Tuple, Dict, Set

from .case import create_case, Case
from ..utils import SubclassJSONSerializer, get_full_class_name, get_type_from_string, conclusion_to_json, is_iterable


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

    def __init__(self, user_input: Optional[str] = None, conclusion_type: Optional[Tuple[Type]] = None,
                 expression_tree: Optional[AST] = None,
                 scope: Optional[Dict[str, Any]] = None, conclusion: Optional[Any] = None):
        """
        Create a callable expression.

        :param user_input: The input given by the expert.
        :param conclusion_type: The type of the output of the callable.
        :param expression_tree: The AST tree parsed from the user input.
        :param scope: The scope to use for the callable expression.
        :param conclusion: The conclusion to use for the callable expression.
        """
        if user_input is None and conclusion is None:
            raise ValueError("Either user_input or conclusion must be provided.")
        self.conclusion: Optional[Any] = conclusion
        self.user_input: str = user_input
        if conclusion_type is not None:
            if is_iterable(conclusion_type):
                conclusion_type = tuple(conclusion_type)
            else:
                conclusion_type = (conclusion_type,)
        self.conclusion_type = conclusion_type
        self.scope: Optional[Dict[str, Any]] = scope if scope is not None else {}
        if conclusion is None:
            self.scope = get_used_scope(self.user_input, self.scope)
            self.expression_tree: AST = expression_tree if expression_tree else parse_string_to_expression(self.user_input)
            self.code = compile_expression_to_code(self.expression_tree)
            self.visitor = VariableVisitor()
            self.visitor.visit(self.expression_tree)

    def __call__(self, case: Any, **kwargs) -> Any:
        try:
            if self.user_input is not None:
                if not isinstance(case, Case):
                    case = create_case(case, max_recursion_idx=3)
                scope = {'case': case, **self.scope}
                output = eval(self.code, scope)
                if output is None:
                    output = scope['_get_value'](case)
                if self.conclusion_type is not None:
                    if is_iterable(output) and not isinstance(output, self.conclusion_type):
                        assert isinstance(list(output)[0], self.conclusion_type), (f"Expected output type {self.conclusion_type},"
                                                                                 f" got {type(output)}")
                    else:
                        assert isinstance(output, self.conclusion_type), (f"Expected output type {self.conclusion_type},"
                                                                          f" got {type(output)}")
                return output
            else:
                return self.conclusion
        except Exception as e:
            raise ValueError(f"Error during evaluation: {e}")

    def combine_with(self, other: 'CallableExpression') -> 'CallableExpression':
        """
        Combine this callable expression with another callable expression using the 'and' operator.
        """
        new_user_input = f"({self.user_input}) and ({other.user_input})"
        return CallableExpression(new_user_input, conclusion_type=self.conclusion_type)

    def __eq__(self, other):
        """
        Check if two callable expressions are equal.
        """
        if not isinstance(other, CallableExpression):
            return False
        return self.user_input == other.user_input and self.conclusion == other.conclusion

    def __hash__(self):
        """
        Hash the callable expression.
        """
        conclusion_hash = self.conclusion if not isinstance(self.conclusion, set) else frozenset(self.conclusion)
        return hash((self.user_input, conclusion_hash))

    def __str__(self):
        """
        Return the user string where each compare is written in a line using compare column offset start and end.
        """
        if self.user_input is None:
            return str(self.conclusion)
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
        return {"user_input": self.user_input,
                "conclusion_type": [get_full_class_name(t) for t in self.conclusion_type]
                if self.conclusion_type is not None else None,
                "scope": {k: get_full_class_name(v) for k, v in self.scope.items()
                          if hasattr(v, '__module__') and hasattr(v, '__name__')},
                "conclusion": conclusion_to_json(self.conclusion),
                }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> CallableExpression:
        return cls(user_input=data["user_input"],
                   conclusion_type=tuple(get_type_from_string(t) for t in data["conclusion_type"])
                   if data["conclusion_type"] else None,
                   scope={k: get_type_from_string(v) for k, v in data["scope"].items()},
                   conclusion=SubclassJSONSerializer.from_json(data["conclusion"]))


def compile_expression_to_code(expression_tree: AST) -> Any:
    """
    Compile an expression tree that was parsed from string into code that can be executed using 'eval(code)'

    :param expression_tree: The parsed expression tree.
    :return: The code that was compiled from the expression tree.
    """
    mode = 'exec' if isinstance(expression_tree, ast.Module) else 'eval'
    return compile(expression_tree, filename="<string>", mode=mode)


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
