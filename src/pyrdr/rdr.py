from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import networkx as nx
from anytree import RenderTree, NodeMixin
from anytree.exporter import DotExporter
from matplotlib import pyplot as plt
from orderedset import OrderedSet
from typing_extensions import List, Optional, Self, Dict

from .datastructures import Category, Attribute, Condition, Case, Stop, MCRDRMode
from .experts import Expert, Human
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

    def __init__(self, conditions: Dict[str, Condition],
                 conclusion: Optional[Category] = None,
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
        if self.conclusion:
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
    expert_accepted_conclusions: Optional[List[Category]] = None
    """
    The conclusions that the expert has accepted, such that they are not asked again.
    """

    @abstractmethod
    def classify(self, x: Case, target: Optional[Category] = None,
                 expert: Optional[Expert] = None, **kwargs) -> Category:
        """
        Classify a case, and ask the user for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category if provided.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        pass

    def fit(self, x_batch: List[Case], y_batch: List[Category],
            expert: Optional[Expert] = None,
            n_iter: int = None,
            draw_tree: bool = False,
            **kwargs_for_classify):
        """
        Fit the classifier to a batch of cases and categories.

        :param x_batch: The batch of cases to fit the classifier to.
        :param y_batch: The batch of categories to fit the classifier to.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param n_iter: The number of iterations to fit the classifier for.
        :param draw_tree: Whether to draw the tree while fitting the classifier.
        """
        if draw_tree:
            plt.ion()
            self.fig = plt.figure()
        all_pred = 0
        i = 0
        while (all_pred != len(y_batch) and n_iter and i < n_iter) \
                or (not n_iter and all_pred != len(y_batch)):
            all_pred = 0
            for x, y in zip(x_batch, y_batch):
                pred_cat = self.classify(x, y, expert=expert, **kwargs_for_classify)
                pred_cat = pred_cat if isinstance(pred_cat, list) else [pred_cat]
                match = y in pred_cat
                if not match:
                    print(f"Predicted: {pred_cat[0]} but expected: {y}")
                all_pred += int(match)
                if draw_tree:
                    self.draw_tree()
                i += 1
                if n_iter and i >= n_iter:
                    break
            print(f"Accuracy: {all_pred}/{len(y_batch)}")
        print(f"Finished training in {i} iterations")
        if draw_tree:
            plt.ioff()
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
                 expert: Optional[Expert] = None, **kwargs) -> Category:
        """
        Classify a case, and ask the user for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category if provided.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        expert = expert if expert else Human()
        if not self.start_rule:
            conditions = expert.ask_for_conditions(x, target)
            self.start_rule = Rule(conditions, target, corner_case=Case(x.id_, list(x.attributes.values())))

        pred = self.start_rule(x)

        if target and pred.conclusion != target:
            conditions = expert.ask_for_conditions(x, target, pred)
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
    evaluated_rules: Optional[List[Rule]] = None
    """
    The evaluated rules in the classifier for one case.
    """
    conclusions: Optional[List[Category]] = None
    """
    The conclusions that the case belongs to.
    """
    stop_rule_conditions: Optional[Dict[str, Condition]] = None
    """
    The conditions of the stopping rule if needed.
    """

    def __init__(self, start_rules: Optional[List[Rule]] = None,
                 mode: MCRDRMode = MCRDRMode.StopOnly):
        """
        :param start_rules: The starting rules for the classifier, these are the rules that are at the top of the tree
        and are always checked, in contrast to the refinement and alternative rules which are only checked if the
        starting rules fire or not.
        :param mode: The mode of the classifier, either StopOnly or StopPlusRule.
        """
        self.start_rules = start_rules
        self.mode: MCRDRMode = mode

    @property
    def start_rule(self):
        """
        Get the starting rule of the classifier.
        """
        return self.start_rules[0] if self.start_rules else None

    def classify(self, x: Case, target: Optional[Category] = None,
                 expert: Optional[Expert] = None, add_extra_conclusions: bool = False) -> List[Category]:
        """
        Classify a case, and ask the user for stopping rules or classifying rules if the classification is incorrect
         or missing by comparing the case with the target category if provided.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions or for extra conclusions.
        :param add_extra_conclusions: Whether to add extra conclusions after classification is done.
        :return: The conclusions that the case belongs to.
        """
        expert = expert if expert else Human()
        if not self.start_rules:
            conditions = expert.ask_for_conditions(x, target)
            self.start_rules = [Rule(conditions, target, corner_case=Case(x.id_, x.attributes_list))]

        rule_idx = 0
        self.evaluated_rules = []
        self.conclusions = []
        self.expert_accepted_conclusions = []
        self.stop_rule_conditions = None
        extra_conclusions = []
        while rule_idx < len(self.start_rules):
            evaluated_rule = self.start_rules[rule_idx](x)

            if (target and evaluated_rule.fired
                    and evaluated_rule.conclusion not in [target, Stop(), *extra_conclusions]):
                conclusions = list(OrderedSet(self.conclusions))
                if evaluated_rule.conclusion not in self.expert_accepted_conclusions:
                    if (add_extra_conclusions and expert.ask_if_conclusion_is_correct(x, evaluated_rule.conclusion,
                                                                                     target=target,
                                                                                     current_conclusions=conclusions)):
                        self.add_conclusion(evaluated_rule)
                        self.expert_accepted_conclusions.append(evaluated_rule.conclusion)
                    else:
                        conditions = expert.ask_for_conditions(x, target, evaluated_rule)
                        evaluated_rule.add_refinement(x, conditions, Stop())
                        if self.mode == MCRDRMode.StopPlusRule:
                            self.stop_rule_conditions = conditions

            elif evaluated_rule.fired and evaluated_rule.conclusion != Stop():
                # Rule fired and target is correct or there is no target to compare
                self.add_conclusion(evaluated_rule)

            if (target and rule_idx >= len(self.start_rules) - 1
                    and target not in self.conclusions):
                # Nothing fired and there is a target that should have fired
                if self.stop_rule_conditions and self.mode == MCRDRMode.StopPlusRule:
                    conditions = self.stop_rule_conditions
                    self.stop_rule_conditions = None
                else:
                    conditions = expert.ask_for_conditions(x, target)
                self.add_top_rule(conditions, target, x)
                rule_idx = 0  # Have to check all rules again to make sure only this new rule fires
            else:
                rule_idx += 1

            if (add_extra_conclusions and not extra_conclusions
                    and target and target in self.conclusions
                    and rule_idx == len(self.start_rules)):
                # Add extra conclusions if needed
                conclusions = list(OrderedSet(self.conclusions))
                print("current conclusions:", conclusions)
                extra_conclusions_dict = expert.ask_for_extra_conclusions(x, conclusions)
                if extra_conclusions_dict:
                    for conclusion, conditions in extra_conclusions_dict.items():
                        self.add_top_rule(conditions, conclusion, x)
                        extra_conclusions.append(conclusion)

        return list(OrderedSet(self.conclusions))

    def add_conclusion(self, evaluated_rule: Rule):
        """
        Add the conclusion of the evaluated rule to the list of conclusions.
        """
        self.evaluated_rules.append(evaluated_rule)
        self.conclusions.append(evaluated_rule.conclusion)

    def add_top_rule(self, conditions: Dict[str, Condition], conclusion: Category, corner_case: Case):
        """
        Add a top rule to the classifier, which is a rule that is always checked and is part of the start_rules list.

        :param conditions: The conditions of the rule.
        :param conclusion: The conclusion of the rule.
        :param corner_case: The corner case of the rule.
        """
        self.start_rules.append(self.start_rules[-1].add_connected_rule(conditions, conclusion, corner_case,
                                                                        edge_weight="next"))


class GeneralRDR(RippleDownRules):
    """
    A general ripple down rules classifier, which can draw multiple conclusions for a case, but each conclusion is part
    of a set of mutually exclusive conclusions. Whenever a conclusion is made, the classification restarts from the
    starting rule, and all the rules that belong to the class of the made conclusion are not checked again. This
    continues until no more rules can be fired. In addition, previous conclusions can be used as conditions or input to
    the next classification/cycle.
    Another possible mode is to have rules that are considered final, when fired, inference will not be restarted,
     and only a refinement can be made to the final rule, those can also be used in another SCRD of their own that
     gets called when the final rule fires.
    """
    def classify(self, x: Case, target: Optional[Category] = None,
                 expert: Optional[Expert] = None,
                 **kwargs) -> Category:
        pass
