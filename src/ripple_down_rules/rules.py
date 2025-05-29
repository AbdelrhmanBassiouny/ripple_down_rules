from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from uuid import uuid4

from anytree import NodeMixin
from sqlalchemy.orm import DeclarativeBase as SQLTable
from typing_extensions import List, Optional, Self, Union, Dict, Any, Tuple

from .datastructures.callable_expression import CallableExpression
from .datastructures.case import Case
from .datastructures.enums import RDREdge, Stop
from .utils import SubclassJSONSerializer, conclusion_to_json, get_full_class_name


class Rule(NodeMixin, SubclassJSONSerializer, ABC):
    fired: Optional[bool] = None
    """
    Whether the rule has fired or not.
    """

    def __init__(self, conditions: Optional[CallableExpression] = None,
                 conclusion: Optional[CallableExpression] = None,
                 parent: Optional[Rule] = None,
                 corner_case: Optional[Union[Case, SQLTable]] = None,
                 weight: Optional[str] = None,
                 conclusion_name: Optional[str] = None,
                 uid: Optional[str] = None):
        """
        A rule in the ripple down rules classifier.

        :param conditions: The conditions of the rule.
        :param conclusion: The conclusion of the rule when the conditions are met.
        :param parent: The parent rule of this rule.
        :param corner_case: The corner case that this rule is based on/created from.
        :param weight: The weight of the rule, which is the type of edge connecting the rule to its parent.
        :param conclusion_name: The name of the conclusion of the rule.
        :param uid: The unique id of the rule.
        """
        super(Rule, self).__init__()
        self.conclusion = conclusion
        self.corner_case = corner_case
        self.parent = parent
        self.weight: Optional[str] = weight
        self.conditions = conditions if conditions else None
        self.conclusion_name: Optional[str] = conclusion_name
        self.json_serialization: Optional[Dict[str, Any]] = None
        self._name: Optional[str] = None
        # generate a unique id for the rule using uuid4
        self.uid: str = uid if uid else str(uuid4().int)

    def _post_detach(self, parent):
        """
        Called after this node is detached from the tree, useful when drawing the tree.

        :param parent: The parent node from which this node was detached.
        """
        self.weight = None

    def __call__(self, x: Case) -> Self:
        return self.evaluate(x)

    def evaluate(self, x: Case) -> Rule:
        """
        Check if the rule or its refinement or its alternative match the case,
        by checking if the conditions are met, then return the rule that matches the case.

        :param x: The case to evaluate the rule on.
        :return: The rule that fired or the last evaluated rule if no rule fired.
        """
        if not self.conditions:
            raise ValueError("Rule has no conditions")
        self.fired = self.conditions(x)
        return self.evaluate_next_rule(x)

    @abstractmethod
    def evaluate_next_rule(self, x: Case):
        """
        Evaluate the next rule after this rule is evaluated.
        """
        pass

    def write_conclusion_as_source_code(self, parent_indent: str = "", defs_file: Optional[str] = None) -> str:
        """
        Get the source code representation of the conclusion of the rule.

        :param parent_indent: The indentation of the parent rule.
        :param defs_file: The file to write the conclusion to if it is a definition.
        :return: The source code representation of the conclusion of the rule.
        """
        if self.conclusion.user_input is not None:
            conclusion = self.conclusion.user_input
        else:
            conclusion = self.conclusion.conclusion
        conclusion_func, conclusion_func_call = self.get_conclusion_as_source_code(conclusion,
                                                                                   parent_indent=parent_indent)
        if conclusion_func is not None:
            with open(defs_file, 'a') as f:
                f.write(conclusion_func.strip() + "\n\n\n")
        return conclusion_func_call

    def get_conclusion_as_source_code(self, conclusion: Any, parent_indent: str = "") -> Tuple[Optional[str], str]:
        """
        Convert the conclusion of a rule to source code.

        :param conclusion: The conclusion to convert to source code.
        :param parent_indent: The indentation of the parent rule.
        :return: The source code of the conclusion as a tuple of strings, one for the function and one for the call.
        """
        if "def " in conclusion:
            # This means the conclusion is a definition that should be written and then called
            conclusion_lines = conclusion.split('\n')
            # use regex to replace the function name
            new_function_name = f"def conclusion_{self.uid}"
            conclusion_lines[0] = re.sub(r"def (\w+)", new_function_name, conclusion_lines[0])
            func_call = f"{parent_indent}    return {new_function_name.replace('def ', '')}(case)\n"
            return "\n".join(conclusion_lines).strip(' '), func_call
        else:
            raise ValueError(f"Conclusion format is not valid, it should contain a function definition."
                             f" Instead got:\n{conclusion}\n")

    def write_condition_as_source_code(self, parent_indent: str = "", defs_file: Optional[str] = None) -> str:
        """
        Get the source code representation of the conditions of the rule.

        :param parent_indent: The indentation of the parent rule.
        :param defs_file: The file to write the conditions to if they are a definition.
        """
        if_clause = self._if_statement_source_code_clause()
        if "def " in self.conditions.user_input:
            if defs_file is None:
                raise ValueError("Cannot write conditions to source code as definitions python file was not given.")
            # This means the conditions are a definition that should be written and then called
            conditions_lines = self.conditions.user_input.split('\n')
            # use regex to replace the function name
            new_function_name = f"def conditions_{self.uid}"
            conditions_lines[0] = re.sub(r"def (\w+)", new_function_name, conditions_lines[0])
            def_code = "\n".join(conditions_lines)
            with open(defs_file, 'a') as f:
                f.write(def_code.strip() + "\n\n\n")
            return f"\n{parent_indent}{if_clause} {new_function_name.replace('def ', '')}(case):\n"
        else:
            raise ValueError(f"Conditions format is not valid, it should contain a function definition"
                             f" Instead got:\n{self.conditions.user_input}\n")

    @abstractmethod
    def _if_statement_source_code_clause(self) -> str:
        pass

    def _to_json(self) -> Dict[str, Any]:
        try:
            corner_case = SubclassJSONSerializer.to_json_static(self.corner_case) if self.corner_case else None
        except Exception as e:
            logging.debug("Failed to serialize corner case to json, setting it to None. Error: %s", e)
            corner_case = None
        json_serialization = {"_type": get_full_class_name(type(self)),
                              "conditions": self.conditions.to_json(),
                              "conclusion": conclusion_to_json(self.conclusion),
                              "parent": self.parent.json_serialization if self.parent else None,
                              "corner_case": corner_case,
                              "conclusion_name": self.conclusion_name,
                              "weight": self.weight,
                              "uid": self.uid}
        return json_serialization

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Rule:
        try:
            corner_case = Case.from_json(data["corner_case"])
        except Exception as e:
            logging.debug("Failed to load corner case from json, setting it to None.")
            corner_case = None
        loaded_rule = cls(conditions=CallableExpression.from_json(data["conditions"]),
                          conclusion=CallableExpression.from_json(data["conclusion"]),
                          parent=cls.from_json(data["parent"]),
                          corner_case=corner_case,
                          conclusion_name=data["conclusion_name"],
                          weight=data["weight"],
                          uid=data["uid"])
        return loaded_rule

    @property
    def name(self):
        """
        Get the name of the rule, which is the conditions and the conclusion.
        """
        return self._name if self._name is not None else self.__str__()

    @name.setter
    def name(self, new_name: str):
        """
        Set the name of the rule.
        """
        self._name = new_name

    def __str__(self, sep="\n"):
        """
        Get the string representation of the rule, which is the conditions and the conclusion.
        """
        return f"{self.conditions}{sep}=> {self.conclusion}"

    def __repr__(self):
        return self.__str__()


class HasAlternativeRule:
    """
    A mixin class for rules that have an alternative rule.
    """
    _alternative: Optional[Rule] = None
    """
    The alternative rule of the rule, which is evaluated when the rule doesn't fire.
    """
    furthest_alternative: Optional[List[Rule]] = None
    """
    The furthest alternative rule of the rule, which is the last alternative rule in the chain of alternative rules.
    """
    all_alternatives: Optional[List[Rule]] = None
    """
    All alternative rules of the rule, which is all the alternative rules in the chain of alternative rules.
    """

    @property
    def alternative(self) -> Optional[Rule]:
        return self._alternative

    @alternative.setter
    def alternative(self, new_rule: Rule):
        """
        Set the alternative rule of the rule. It is important that no rules should be retracted or changed,
        only new rules should be added.
        """
        if new_rule is None:
            return
        if self.furthest_alternative:
            self.furthest_alternative[-1].alternative = new_rule
        else:
            new_rule.parent = self
            new_rule.weight = RDREdge.Alternative.value if not new_rule.weight else new_rule.weight
            self._alternative = new_rule
        self.furthest_alternative = [new_rule]


class HasRefinementRule:
    _refinement: Optional[HasAlternativeRule] = None
    """
    The refinement rule of the rule, which is evaluated when the rule fires.
    """

    @property
    def refinement(self) -> Optional[Rule]:
        return self._refinement

    @refinement.setter
    def refinement(self, new_rule: Rule):
        """
        Set the refinement rule of the rule. It is important that no rules should be retracted or changed,
        only new rules should be added.
        """
        if new_rule is None:
            return
        new_rule.top_rule = self
        if self.refinement:
            self.refinement.alternative = new_rule
        else:
            new_rule.parent = self
            new_rule.weight = RDREdge.Refinement.value
            self._refinement = new_rule


class SingleClassRule(Rule, HasAlternativeRule, HasRefinementRule):
    """
    A rule in the SingleClassRDR classifier, it can have a refinement or an alternative rule or both.
    """

    def evaluate_next_rule(self, x: Case) -> SingleClassRule:
        if self.fired:
            returned_rule = self.refinement(x) if self.refinement else self
        else:
            returned_rule = self.alternative(x) if self.alternative else self
        return returned_rule if returned_rule.fired else self

    def fit_rule(self, x: Case, target: CallableExpression, conditions: CallableExpression):
        new_rule = SingleClassRule(conditions, target,
                                   corner_case=x, parent=self)
        if self.fired:
            self.refinement = new_rule
        else:
            self.alternative = new_rule

    def _to_json(self) -> Dict[str, Any]:
        self.json_serialization = {**super(SingleClassRule, self)._to_json(),
                                   "refinement": self.refinement.to_json() if self.refinement is not None else None,
                                   "alternative": self.alternative.to_json() if self.alternative is not None else None}
        return self.json_serialization

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> SingleClassRule:
        loaded_rule = super(SingleClassRule, cls)._from_json(data)
        loaded_rule.refinement = SingleClassRule.from_json(data["refinement"])
        loaded_rule.alternative = SingleClassRule.from_json(data["alternative"])
        return loaded_rule

    def _if_statement_source_code_clause(self) -> str:
        return "elif" if self.weight == RDREdge.Alternative.value else "if"


class MultiClassStopRule(Rule, HasAlternativeRule):
    """
    A rule in the MultiClassRDR classifier, it can have an alternative rule and a top rule,
    the conclusion of the rule is a Stop category meant to stop the parent conclusion from being made.
    """
    top_rule: Optional[MultiClassTopRule] = None
    """
    The top rule of the rule, which is the nearest ancestor that fired and this rule is a refinement of.
    """

    def __init__(self, *args, **kwargs):
        super(MultiClassStopRule, self).__init__(*args, **kwargs)
        self.conclusion = CallableExpression(conclusion_type=(Stop,), conclusion=Stop.stop)

    def evaluate_next_rule(self, x: Case) -> Optional[Union[MultiClassStopRule, MultiClassTopRule]]:
        if self.fired:
            self.top_rule.fired = False
            return self.top_rule.alternative
        elif self.alternative:
            return self.alternative(x)
        else:
            return self.top_rule.alternative

    def _to_json(self) -> Dict[str, Any]:
        self.json_serialization = {**Rule._to_json(self),
                                   "alternative": self.alternative.to_json() if self.alternative is not None else None}
        return self.json_serialization

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> MultiClassStopRule:
        loaded_rule = super(MultiClassStopRule, cls)._from_json(data)
        # The following is done to prevent re-initialization of the top rule,
        # so the top rule that is already initialized is passed in the data instead of its json serialization.
        loaded_rule.top_rule = data['top_rule']
        if data['alternative'] is not None:
            data['alternative']['top_rule'] = data['top_rule']
        loaded_rule.alternative = MultiClassStopRule.from_json(data["alternative"])
        return loaded_rule

    def get_conclusion_as_source_code(self, conclusion: Any, parent_indent: str = "") -> Tuple[None, str]:
        return None, f"{parent_indent}{' ' * 4}pass\n"

    def _if_statement_source_code_clause(self) -> str:
        return "elif" if self.weight == RDREdge.Alternative.value else "if"


class MultiClassTopRule(Rule, HasRefinementRule, HasAlternativeRule):
    """
    A rule in the MultiClassRDR classifier, it can have a refinement and a next rule.
    """

    def __init__(self, *args, **kwargs):
        super(MultiClassTopRule, self).__init__(*args, **kwargs)
        self.weight = RDREdge.Next.value

    def evaluate_next_rule(self, x: Case) -> Optional[Union[MultiClassStopRule, MultiClassTopRule]]:
        if self.fired and self.refinement:
            return self.refinement(x)
        elif self.alternative:  # Here alternative refers to next rule in MultiClassRDR
            return self.alternative

    def fit_rule(self, x: Case, target: CallableExpression, conditions: CallableExpression):
        if self.fired and target != self.conclusion:
            self.refinement = MultiClassStopRule(conditions, corner_case=x, parent=self)
        elif not self.fired:
            self.alternative = MultiClassTopRule(conditions, target, corner_case=x, parent=self)

    def _to_json(self) -> Dict[str, Any]:
        self.json_serialization = {**Rule._to_json(self),
                                   "refinement": self.refinement.to_json() if self.refinement is not None else None,
                                   "alternative": self.alternative.to_json() if self.alternative is not None else None}
        return self.json_serialization

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> MultiClassTopRule:
        loaded_rule = super(MultiClassTopRule, cls)._from_json(data)
        # The following is done to prevent re-initialization of the top rule,
        # so the top rule that is already initialized is passed in the data instead of its json serialization.
        if data['refinement'] is not None:
            data['refinement']['top_rule'] = loaded_rule
        loaded_rule.refinement = MultiClassStopRule.from_json(data["refinement"])
        loaded_rule.alternative = MultiClassTopRule.from_json(data["alternative"])
        return loaded_rule

    def get_conclusion_as_source_code(self, conclusion: Any, parent_indent: str = "") -> Tuple[str, str]:
        func, func_call = super().get_conclusion_as_source_code(str(conclusion), parent_indent=parent_indent)
        conclusion_str = func_call.replace("return ", "").strip()

        statement = f"{parent_indent}    conclusions.update(make_set({conclusion_str}))\n"
        if self.alternative is None:
            statement += f"{parent_indent}return conclusions\n"
        return func, statement

    def _if_statement_source_code_clause(self) -> str:
        return "if"
