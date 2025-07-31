import itertools

from typing_extensions import Dict, List, TypeVar, Type

from .symbolic import Variable, SymbolicExpression, ConstrainingOperator
from .utils import filter_data


class Generate:
    def __init__(self, *symbolic_variables: Variable):
        self.symbolic_variables = symbolic_variables
        self._yield_single_value = False
        if len(symbolic_variables) == 1:
            self._yield_single_value = True

    def __iter__(self):
        iterables = [iter(sv) for sv in self.symbolic_variables]
        while True:
            try:
                yield next(iterables[0]) if self._yield_single_value else [next(sv_iter) for sv_iter in iterables]
            except StopIteration:
                break

    def where(self, condition: SymbolicExpression):
        """
        Apply condition to filter the generated symbolic variables.

        :param condition: Condition to apply to the generated variables.
        :return: A new Generate instance with the filtered results.
        """
        where(condition)
        return self


def where(*conditions: SymbolicExpression):
    """
    Apply conditions to filter the generated symbolic variables.

    :param conditions: Condition to apply to the generated variables.
    :return: A new Generate instance with the filtered results.
    """
    for condition in conditions:
        if isinstance(condition, ConstrainingOperator):
            condition.constrain_()
        else: # a boolean expression
            condition_indices = (i for i, value in enumerate(condition) if value)
            condition.parent_.constrain(condition_indices)
