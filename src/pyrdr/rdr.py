from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import networkx as nx
from anytree import RenderTree, NodeMixin
from anytree.exporter import DotExporter
from matplotlib import pyplot as plt
from orderedset import OrderedSet
from typing_extensions import List, Optional, Self, Dict, Callable

from .ask_experts import ask_human
from .datastructures import Category, Attribute, Condition, Case, Stop
from .utils import tree_to_graph


class Rule(NodeMixin):
    fired: bool = False
    """
    Whether the rule has fired or not.
    """
    refinement: Optional[Rule] = None
    """
    The refinement rule (called when this rule fires) of this rule if it exists.
    """
    alternative: Optional[Rule] = None
    """
    The alternative rule (called when this rule doesn't fire) of this rule if it exists.
    """
    all_rules: Dict[str, Rule] = {}
    """
    All rules in the classifier.
    """
    rule_idx: int = 0
    """
    The index of the rule in the all rules list.
    """

    def __init__(self, conditions: Dict[str, Condition], conclusion: Category,
                 edge_weight: Optional[str] = None,
                 parent: Optional[Rule] = None,
                 corner_case: Optional[Case] = None):
        """
        A rule in the ripple down rules classifier.

        :param conditions: The conditions of the rule.
        :param conclusion: The conclusion of the rule when the conditions are met.
        :param edge_weight: The weight of the edge to the parent rule if it exists.
        :param parent: The parent rule of this rule.
        :param corner_case: The corner case that this rule is based on/created from.
        """
        super(Rule, self).__init__()
        self.conditions = conditions
        self.corner_case = corner_case
        self.conclusion = conclusion
        self.parent = parent
        self.weight = edge_weight if parent is not None else None
        self._name = self.__str__()
        self.update_all_rules()

    def update_all_rules(self):
        """
        Update the all rules dictionary with this rule. And make the rule name unique if it already exists.
        """
        if self.name in self.all_rules:
            self.all_rules[self.name].append(self)
            self.rule_idx = len(self.all_rules[self.name]) - 1
            self.name += f"_{self.rule_idx}"
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
        for att_name, condition in self.conditions.items():
            if att_name not in x.attributes or not condition(x.attributes[att_name].value):
                self.fired = False
                return self.alternative(x) if self.alternative else self
        self.fired = True
        refined_rule = self.refinement(x) if self.refinement else None
        return refined_rule if refined_rule and refined_rule.fired else self

    def add_alternative(self, x: Case, conditions: Dict[str, Condition], conclusion: Category):
        """
        Add an alternative rule to this rule (called when this rule doesn't fire).

        :param x: The case to add the alternative rule for.
        :param conditions: The conditions of the alternative rule.
        :param conclusion: The conclusion of the alternative rule.
        """
        if self.alternative:
            self.alternative.add_alternative(x, conditions, conclusion)
        else:
            self.alternative = self.add_connected_rule(conditions, conclusion, x, edge_weight="else if")

    def add_refinement(self, x: Case, conditions: Dict[str, Condition], conclusion: Category):
        """
        Add a refinement rule to this rule (called when this rule fires).

        :param x: The case to add the refinement rule for.
        :param conditions: The conditions of the refinement rule.
        :param conclusion: The conclusion of the refinement rule.
        """
        if self.refinement:
            self.refinement.add_alternative(x, conditions, conclusion)
        else:
            self.refinement = self.add_connected_rule(conditions, conclusion, x, edge_weight="except if")

    def add_connected_rule(self, conditions: Dict[str, Condition], conclusion: Category,
                           corner_case: Case,
                           edge_weight: Optional[str] = None) -> Rule:
        """
        Add a connected rule to this rule, connected in the sense that it has this rule as parent.

        :param conditions: The conditions of the connected rule.
        :param conclusion: The conclusion of the connected rule.
        :param corner_case: The corner case of the connected rule.
        :param edge_weight: The weight of the edge to the parent rule.
        :return: The connected rule.
        """
        return Rule(conditions, conclusion, corner_case=Case(corner_case.id_, list(corner_case.attributes.values())),
                    parent=self, edge_weight=edge_weight)

    def get_different_attributes(self, x: Case) -> Dict[str, Attribute]:
        """
        Get the differentiating attributes between the corner case and the case.

        :param x: The case to compare with the corner case.
        :return: The differentiating attributes between the corner case and the case as a dictionary.
        """
        return {a.name: a for a in self.corner_case.attributes.values()
                if a not in x.attributes.values()}

    @property
    def name(self):
        """
        Get the name of the rule, which is the conditions and the conclusion.
        """
        return self._name

    @name.setter
    def name(self, value):
        """
        Set the name of the rule.
        """
        self._name = value

    def __str__(self, sep="\n"):
        """
        Get the string representation of the rule, which is the conditions and the conclusion.
        """
        conditions = f"^{sep}".join([str(c) for c in list(self.conditions.values())])
        conditions += f"{sep}=> {self.conclusion.name}"
        return conditions

    def __repr__(self):
        return self.__str__()


class RippleDownRules(ABC):
    """
    The abstract base class for the ripple down rules classifiers.
    """
    start_rule: Optional[Rule] = None
    """
    The starting rule for the classifier tree.
    """
    fig: Optional[plt.Figure] = None
    """
    The figure to draw the tree on.
    """
    all_expert_answers: Optional[List[Dict[str, Condition]]] = None
    """
    All expert answers given by the expert function.
    """

    @abstractmethod
    def classify(self, x: Case, target: Optional[Category] = None,
                 ask_expert: Optional[Callable] = None) -> Category:
        """
        Classify a case, and ask the user for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category if provided.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :param ask_expert: The expert function to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        pass

    def fit(self, x_batch: List[Case], y_batch: List[Category],
            ask_expert: Optional[Callable] = None,
            n_iter: int = None):
        """
        Fit the classifier to a batch of cases and categories.

        :param x_batch: The batch of cases to fit the classifier to.
        :param y_batch: The batch of categories to fit the classifier to.
        :param ask_expert: The expert function to ask for differentiating features as new rule conditions.
        :param n_iter: The number of iterations to fit the classifier for.
        """
        plt.ion()
        # Start tree updating in a separate thread
        self.fig = plt.figure()
        self.all_expert_answers = []
        all_pred = 0
        i = 0
        while (all_pred != len(y_batch) and n_iter and i < n_iter) \
                or (not n_iter and all_pred != len(y_batch)):
            all_pred = 0
            for x, y in zip(x_batch, y_batch):
                pred_cat = self.classify(x, y, ask_expert=ask_expert)
                pred_cat = pred_cat if isinstance(pred_cat, list) else [pred_cat]
                match = len(pred_cat) == 1 and pred_cat[0] == y
                if not match:
                    print(f"Predicted: {pred_cat[0]} but expected: {y}")
                all_pred += int(match)
                self.draw_tree()
                i += 1
                if n_iter and i >= n_iter:
                    break
            print(f"Accuracy: {all_pred}/{len(y_batch)}")
            print("all expert answers:", self.all_expert_answers)
        plt.ioff()
        print(f"Finished training in {i} iterations")
        plt.show()

    @staticmethod
    def edge_attr_setter(parent, child):
        """
        Set the edge attributes for the dot exporter.
        """
        if child is None or child.weight is None:
            return ""
        return f'style="bold", label=" {child.weight}"'

    def render_tree(self, use_dot_exporter: bool = False,
                    filename: str = "scrdr"):
        """
        Render the tree using the console and optionally export it to a dot file.

        :param use_dot_exporter: Whether to export the tree to a dot file.
        :param filename: The name of the file to export the tree to.
        """
        if not self.start_rule:
            logging.warning("No rules to render")
            return
        for pre, _, node in RenderTree(self.start_rule):
            print(f"{pre}{node.weight or ''} {node.__str__(sep='')}")
        if use_dot_exporter:
            de = DotExporter(self.start_rule,
                             edgeattrfunc=self.edge_attr_setter
                             )
            de.to_dotfile(f"{filename}{'.dot'}")
            de.to_picture(f"{filename}{'.png'}")

    def draw_tree(self):
        """
        Draw the tree using matplotlib and networkx.
        """
        if self.start_rule is None:
            return
        self.fig.clf()
        graph = tree_to_graph(self.start_rule)
        fig_sz_x = 13
        fig_sz_y = 10
        self.fig.set_size_inches(fig_sz_x, fig_sz_y)
        pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot")
        # scale down pos
        max_pos_x = max([v[0] for v in pos.values()])
        max_pos_y = max([v[1] for v in pos.values()])
        pos = {k: (v[0] * fig_sz_x / max_pos_x, v[1] * fig_sz_y / max_pos_y) for k, v in pos.items()}
        nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1000,
                ax=self.fig.gca(), node_shape="o", font_size=8)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'),
                                     ax=self.fig.gca(), rotate=False, clip_on=False)
        plt.pause(0.1)


class SingleClassRDR(RippleDownRules):

    def __init__(self, start_rule: Optional[Rule] = None):
        """
        A single class ripple down rule classifier.

        :param start_rule: The starting rule for the classifier.
        """
        self.start_rule = start_rule
        self.fig: Optional[plt.Figure] = None

    def classify(self, x: Case, target: Optional[Category] = None,
                 ask_expert: Optional[Callable] = None) -> Category:
        """
        Classify a case, and ask the user for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category if provided.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :param ask_expert: The expert function to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        ask_expert = ask_expert if ask_expert else ask_human
        if not self.start_rule:
            conditions = ask_expert(x, target)
            self.start_rule = Rule(conditions, target, corner_case=Case(x.id_, list(x.attributes.values())))

        pred = self.start_rule(x)

        if target and pred.conclusion != target:
            diff_attributes = pred.get_different_attributes(x)
            conditions = ask_expert(x, target, pred, diff_attributes)
            if pred.fired:
                pred.add_refinement(x, conditions, target)
            else:
                pred.add_alternative(x, conditions, target)

        return pred.conclusion


class MultiClassRDR(RippleDownRules):
    """
    A multi class ripple down rules classifier, which can draw multiple conclusions for a case.
    This is done by going through all rules and checking if they fire or not, and adding stopping rules if needed,
    when wrong conclusions are made to stop these rules from firing again for similar cases.
    """
    def __init__(self, start_rules: Optional[List[Rule]] = None):
        """
        :param start_rules: The starting rules for the classifier, these are the rules that are at the top of the tree
        and are always checked, in contrast to the refinement and alternative rules which are only checked if the
        starting rules fire or not.
        """
        self.start_rules = start_rules

    @property
    def start_rule(self):
        """
        Get the starting rule of the classifier.
        """
        return self.start_rules[0] if self.start_rules else None

    def classify(self, x: Case, target: Optional[Category] = None,
                 ask_expert: Optional[Callable] = None) -> List[Category]:
        """
        Classify a case, and ask the user for stopping rules or classifying rules if the classification is incorrect
         or missing by comparing the case with the target category if provided.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :param ask_expert: The expert function to ask for differentiating features as new rule conditions.
        :return: The conclusions that the case belongs to.
        """
        ask_expert = ask_expert if ask_expert else ask_human
        self.all_expert_answers = [] if not self.all_expert_answers else self.all_expert_answers
        if not self.start_rules:
            conditions = ask_expert(x, target)
            self.all_expert_answers.append(conditions)
            self.start_rules = [Rule(conditions, target, corner_case=Case(x.id_, list(x.attributes.values())))]

        rule_idx = 0
        evaluated_rules = []
        conclusions = []
        stop_rule_conditions = None
        while rule_idx < len(self.start_rules):
            evaluated_rule = self.start_rules[rule_idx](x)

            if target and evaluated_rule.fired and evaluated_rule.conclusion not in [target, Stop()]:
                diff_attributes = evaluated_rule.get_different_attributes(x)
                conditions = ask_expert(x, target, evaluated_rule, diff_attributes)
                self.all_expert_answers.append(conditions)
                evaluated_rule.add_refinement(x, conditions, Stop())
                stop_rule_conditions = conditions

            elif evaluated_rule.fired and evaluated_rule.conclusion != Stop():  # Rule fired and target is correct or there is no target to compare
                stop_rule_conditions = None
                evaluated_rules.append(evaluated_rule)
                conclusions.append(evaluated_rule.conclusion)

            if (target and rule_idx >= len(self.start_rules) - 1
                    and target not in conclusions):
                # Nothing fired and there is a target that should have fired
                if stop_rule_conditions:
                    conditions = stop_rule_conditions
                    stop_rule_conditions = None
                else:
                    conditions = ask_expert(x, target)
                    self.all_expert_answers.append(conditions)
                self.add_top_rule(conditions, target, x)
                rule_idx = 0  # Have to check all rules again to make sure only this new rule fires
                continue
            rule_idx += 1
        return list(OrderedSet(conclusions))

    def add_top_rule(self, conditions: Dict[str, Condition], conclusion: Category, corner_case: Case):
        """
        Add a top rule to the classifier, which is a rule that is always checked and is part of the start_rules list.

        :param conditions: The conditions of the rule.
        :param conclusion: The conclusion of the rule.
        :param corner_case: The corner case of the rule.
        """
        self.start_rules.append(self.start_rules[-1].add_connected_rule(conditions, conclusion, corner_case,
                                                                        edge_weight="next"))
