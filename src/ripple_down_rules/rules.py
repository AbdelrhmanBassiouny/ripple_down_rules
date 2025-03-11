from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from anytree import NodeMixin
from typing_extensions import List, Optional, Self, Union, Dict, Any

from .datastructures import CallableExpression, Case, SQLTable
from .datastructures.enums import RDREdge, Stop
from .utils import SubclassJSONSerializer


class Rule(NodeMixin, SubclassJSONSerializer, ABC):
    fired: Optional[bool] = None
    """
    Whether the rule has fired or not.
    """

    def __init__(self, conditions: Optional[CallableExpression] = None,
                 conclusion: Optional[CallableExpression] = None,
                 parent: Optional[Rule] = None,
                 corner_case: Optional[Union[Case, SQLTable]] = None,
                 weight: Optional[str] = None):
        """
        A rule in the ripple down rules classifier.

        :param conditions: The conditions of the rule.
        :param conclusion: The conclusion of the rule when the conditions are met.
        :param parent: The parent rule of this rule.
        :param corner_case: The corner case that this rule is based on/created from.
        :param weight: The weight of the rule, which is the type of edge connecting the rule to its parent.
        """
        super(Rule, self).__init__()
        self.conclusion = conclusion
        self.corner_case = corner_case
        self.parent = parent
        self.weight: Optional[str] = weight
        self.conditions = conditions if conditions else None

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

    @property
    def name(self):
        """
        Get the name of the rule, which is the conditions and the conclusion.
        """
        return self.__str__()

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

    def write_conclusion_as_source_code(self, parent_indent: str = "") -> str:
        """
        Get the source code representation of the conclusion of the rule.

        :param parent_indent: The indentation of the parent rule.
        """
        if isinstance(self.conclusion, CallableExpression):
            conclusion = self.conclusion.parsed_user_input
        elif isinstance(self.conclusion, Enum):
            conclusion = str(self.conclusion)
        else:
            conclusion = self.conclusion
        return f"{parent_indent}{' ' * 4}return {conclusion}\n"

    def write_condition_as_source_code(self, parent_indent: str = "") -> str:
        """
        Get the source code representation of the conditions of the rule.

        :param parent_indent: The indentation of the parent rule.
        """
        if_clause = "elif" if self.weight == RDREdge.Alternative.value else "if"
        return f"{parent_indent}{if_clause} {self.conditions.parsed_user_input}:\n"

    def to_json(self) -> Dict[str, Any]:
        return {**SubclassJSONSerializer.to_json(self),
                "conditions": self.conditions.to_json(),
                "conclusion": self.conclusion.to_json(),
                "parent": self.parent.to_json() if self.parent else None,
                "corner_case": self.corner_case.to_json() if self.corner_case else None,
                "weight": self.weight}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(conditions=CallableExpression.from_json(data["conditions"]),
                   conclusion=CallableExpression.from_json(data["conclusion"]),
                   parent=SingleClassRule.from_json(data["parent"]),
                   corner_case=Case.from_json(data["corner_case"]),
                   weight=data["weight"])




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
        self.conclusion = Stop.stop

    def evaluate_next_rule(self, x: Case) -> Optional[Union[MultiClassStopRule, MultiClassTopRule]]:
        if self.fired:
            self.top_rule.fired = False
            return self.top_rule.alternative
        elif self.alternative:
            return self.alternative(x)
        else:
            return self.top_rule.alternative


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
