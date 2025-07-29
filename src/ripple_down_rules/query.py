import itertools

from typing_extensions import Dict, List, TypeVar, Type

from .symbolic import Variable, SymbolicExpression
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
        for item, values in condition.variables_data_dict.items():
            indices = list(values)
            if len(indices) > 0 and type(indices[0]) == bool:
                indices = [i for i, v in enumerate(indices) if v]
            item.data = filter_data(item.data, indices)
        return Generate(*self.symbolic_variables)


def where(condition: SymbolicExpression):
    """
    Apply condition to filter the generated symbolic variables.

    :param condition: Condition to apply to the generated variables.
    :return: A new Generate instance with the filtered results.
    """
    for item, values in condition.variables_data_dict.items():
        indices = list(values)
        if len(indices) > 0 and type(indices[0]) == bool:
            indices = [i for i, v in enumerate(indices) if v]
        item.data = filter_data(item.data, indices)
