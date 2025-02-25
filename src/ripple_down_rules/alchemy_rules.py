from dataclasses import dataclass

import sqlalchemy
from sqlalchemy import Select, BinaryExpression, select
from anytree import NodeMixin
from typing_extensions import Optional, Tuple, Any

from .datastructures import ObjectPropertyTarget, Condition
from .utils import get_prompt_session_for_obj, prompt_and_parse_user_for_input


@dataclass
class Target:
    column: sqlalchemy.Column
    value: Any


def prompt_for_alchemy_conditions(x: sqlalchemy.orm.DeclarativeBase, target: ObjectPropertyTarget,
                                     user_input: Optional[str] = None) -> BinaryExpression:
    """
    Prompt the user for relational conditions.

    :param x: The case to classify.
    :param target: The target category to compare the case with.
    :param user_input: The user input to parse. If None, the user is prompted for input.
    :return: The differentiating features as new rule conditions.
    """
    session = get_prompt_session_for_obj(x)
    prompt_str = f"Give Conditions for {x.__tablename__}.{target.name}"
    user_input, tree = prompt_and_parse_user_for_input(prompt_str, session, user_input=user_input)
    result = eval(user_input)
    return result, target

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




