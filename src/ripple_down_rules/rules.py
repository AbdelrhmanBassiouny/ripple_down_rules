from __future__ import annotations

from abc import ABC, abstractmethod

from anytree import NodeMixin
from typing_extensions import List, Optional, Self, Dict, Union

from .datastructures import Category, Attribute, Condition, Case, Stop, RDREdge


class Rule(NodeMixin, ABC):
    fired: Optional[bool] = None
    """
    Whether the rule has fired or not.
    """
    all_rules: Dict[str, Rule] = {}
    """
    All rules in the classifier.
    """
    rule_idx: int = 0
    """
    The index of the rule in the all rules list.
    """

    def __init__(self, conditions: Optional[Dict[str, Condition]] = None,
                 conclusion: Optional[Category] = None,
                 parent: Optional[Rule] = None,
                 corner_case: Optional[Case] = None,
                 weight: Optional[str] = None):
        """
        A rule in the ripple down rules classifier.

        :param conditions: The conditions of the rule.
        :param conclusion: The conclusion of the rule when the conditions are met.
        :param parent: The parent rule of this rule.
        :param corner_case: The corner case that this rule is based on/created from.
        """
        super(Rule, self).__init__()
        self._conclusion = conclusion
        self.corner_case = corner_case
        self.parent = parent
        self.weight: Optional[str] = weight
        self._conditions = conditions if conditions else {}
        self.update_all_rules()

    @property
    def conditions(self) -> Dict[str, Condition]:
        return self._conditions

    @conditions.setter
    def conditions(self, value: Dict[str, Condition]):
        self._conditions = value
        self.update_all_rules()

    @property
    def conclusion(self) -> Category:
        return self._conclusion

    @conclusion.setter
    def conclusion(self, value: Category):
        self._conclusion = value
        self.update_all_rules()

    def update_all_rules(self):
        """
        Update the all rules dictionary with this rule. And make the rule name unique if it already exists.
        """
        if self.name in self.all_rules:
            self.all_rules[self.name].append(self)
            self.rule_idx = len(self.all_rules[self.name]) - 1
        else:
            self.all_rules[self.name] = [self]

    def _post_detach(self, parent):
        """
        Called after this node is detached from the tree, useful when drawing the tree.

        :param parent: The parent node from which this node was detached.
        """
        self.weight = None

    def __call__(self, x: Case) -> Self:
        return self.evaluate(x)

    def __getitem__(self, attribute_name):
        return self.conditions.get(attribute_name, None)

    def evaluate(self, x: Case) -> Rule:
        """
        Check if the rule or its refinement or its alternative match the case,
        by checking if the conditions are met, then return the rule that matches the case.

        :param x: The case to evaluate the rule on.
        :return: The rule that fired or the last evaluated rule if no rule fired.
        """
        if not self.conditions:
            raise ValueError("Rule has no conditions")
        self.fired = True
        for att_name, condition in self.conditions.items():
            if att_name not in x.attributes or not condition(x.attributes[att_name].value):
                self.fired = False
                break
        return self.evaluate_next_rule(x)

    @abstractmethod
    def evaluate_next_rule(self, x: Case):
        """
        Evaluate the next rule after this rule is evaluated.
        """
        pass

    def get_different_attributes(self, x: Case) -> Dict[str, Attribute]:
        """
        :param x: The case to compare with the corner case.
        :return: The differentiating attributes between the case and the corner case as a dictionary.
        """
        return x - self.corner_case

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
        conditions = f"^{sep}".join([str(c) for c in list(self.conditions.values())])
        if self.conclusion:
            conditions += f"{sep}=> {self.conclusion.name}"
        conditions += f"_{self.rule_idx}" if self.rule_idx > 1 else ""
        return conditions

    def __repr__(self):
        return self.__str__()


class StopRule(Rule, ABC):
    """
    A stopping rule that is used to stop the parent conclusion from being made, thus giving no conclusion instead,
    which is useful to prevent a conclusion in certain condition if it is wrong when these conditions are met.
    """
    conclusion: Category = Stop()
    """
    The conclusion of the stopping rule, which is a Stop category.
    """

    def __init__(self, conditions: Dict[str, Condition], corner_case: Optional[Case] = None,
                 parent: Optional[Rule] = None):
        super(StopRule, self).__init__(conditions, self.conclusion,
                                       corner_case=corner_case, parent=parent)


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

    def fit_rule(self, x: Case, target: Category, conditions: Optional[Dict[str, Condition]] = None):
        if not conditions:
            conditions = Condition.from_two_cases(self.corner_case, x)
        new_rule = SingleClassRule(conditions, target,
                                   corner_case=x, parent=self)
        if self.fired:
            self.refinement = new_rule
        else:
            self.alternative = new_rule


class HasConclusions:
    all_conclusions: List[Category] = []
    """
    All conclusions in the classifier, this gets updated when a new conclusion is added by a fired rule.
    """


class MultiClassStopRule(Rule, HasConclusions, HasAlternativeRule):
    """
    A rule in the MultiClassRDR classifier, it can have an alternative rule and a top rule,
    the conclusion of the rule is a Stop category meant to stop the parent conclusion from being made.
    """
    top_rule: Optional[TopRule] = None
    """
    The top rule of the rule, which is the nearest ancestor that fired and this rule is a refinement of.
    """
    def __init__(self, *args, **kwargs):
        super(MultiClassStopRule, self).__init__(*args, **kwargs)
        self.conclusion = Stop()

    def evaluate_next_rule(self, x: Case) -> HasConclusions:
        if self.fired:
            self.top_rule.fired = False
            return self.top_rule.alternative
        elif self.alternative:
            return self.alternative(x)
        else:
            self.all_conclusions.append(self.top_rule.conclusion)
            return self.top_rule.alternative


class TopRule(Rule, HasConclusions, HasRefinementRule, HasAlternativeRule):
    """
    A rule in the MultiClassRDR classifier, it can have a refinement and a next rule.
    """

    def __init__(self, *args, **kwargs):
        super(TopRule, self).__init__(*args, **kwargs)
        self.weight = RDREdge.Next.value

    def evaluate_next_rule(self, x: Case) -> Optional[Union[MultiClassStopRule, TopRule]]:
        if self.fired:
            if self.refinement:
                return self.refinement(x)
            else:
                self.all_conclusions.append(self.conclusion)
        if self.alternative:
            return self.alternative

    def fit_rule(self, x: Case, target: Category, conditions: Optional[Dict[str, Condition]] = None):
        if not conditions:
            conditions = Condition.from_two_cases(self.corner_case, x)

        if self.fired and target != self.conclusion:
            self.refinement = MultiClassStopRule(conditions, corner_case=x, parent=self)
        elif not self.fired:
            self.alternative = TopRule(conditions, target, corner_case=x, parent=self)
