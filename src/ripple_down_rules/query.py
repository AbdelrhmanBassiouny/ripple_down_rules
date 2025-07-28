import itertools

from .symbolic_variable import SymbolicVariable, SymbolicExpression
from .utils import filter_data


class Generate:
    def __init__(self, *symbolic_variables: SymbolicVariable):
        self.symbolic_variables = symbolic_variables

    def __iter__(self):
        iterables = [iter(sv) for sv in self.symbolic_variables]
        while True:
            try:
                yield [next(sv_iter) for sv_iter in iterables]
            except StopIteration:
                break

    def where(self, *conditions: SymbolicExpression):
        """
        Apply conditions to filter the generated symbolic variables.

        :param conditions: Conditions to apply to the generated variables.
        :return: A new Generate instance with the filtered results.
        """
        for condition in conditions:
            for item, values in condition.variables_data_dict.items():
                indices = list(values)
                item.data = filter_data(item.data, indices)
        return Generate(*self.symbolic_variables)

