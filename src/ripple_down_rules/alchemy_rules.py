from dataclasses import dataclass

import sqlalchemy
from sqlalchemy import Select, BinaryExpression, select
from anytree import NodeMixin
from typing_extensions import Optional, Tuple, Any

from .datastructures import ObjectAttributeTarget, Condition
from .utils import get_prompt_session_for_obj, prompt_user_input_and_parse_to_expression


@dataclass
class Target:
    column: sqlalchemy.Column
    value: Any


class AlchemyRule(NodeMixin):
    statement: BinaryExpression
    target: Target

    def __init__(self, statement: Select, target: Target, parent=None, children=None):
        self.statement = statement
        self.target = target
        self.parent = parent
        if children:
            self.children = children



    def statement_of_path(self):
        statement = select(self.target.column.table)
        for i in self.path[:-1]:
            statement = statement.where(not i.statement)
        statement = statement.where(self.statement)
        return statement




