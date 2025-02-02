from __future__ import annotations

import logging

import networkx as nx
from anytree import RenderTree, NodeMixin
from anytree.exporter import DotExporter
from matplotlib import pyplot as plt
from typing_extensions import List, Optional, Self, Dict, Callable

from .ask_experts import ask_human
from .datastructures import Category, Attribute, Condition, Case
from .utils import tree_to_graph


class Rule(NodeMixin):
    conclusion: Category
    fired: bool = False
    corner_case: Optional[Case] = None
    refinement: Optional[Rule] = None
    alternative: Optional[Rule] = None
    all_rules: Dict[str, Rule] = {}
    rule_idx: int = 0

    def __init__(self, conditions: Dict[str, Condition], category: Category,
                 edge_weight: Optional[str] = None,
                 parent: Optional[Rule] = None,
                 corner_case: Optional[Case] = None):
        super(Rule, self).__init__()
        self.conditions = conditions
        self.corner_case = corner_case
        self.conclusion = category
        self.parent = parent
        self.weight = edge_weight if parent is not None else None
        self._name = self.__str__()
        self.update_all_rules()

    def update_all_rules(self):
        if self.name in self.all_rules:
            self.all_rules[self.name].append(self)
            self.rule_idx = len(self.all_rules[self.name]) - 1
            self.name += f"_{self.rule_idx}"
        else:
            self.all_rules[self.name] = [self]

    def _post_detach(self, parent):
        """
        Called after this node is detached from the tree.

        :param parent: The parent node from which this node was detached.
        """
        self.weight = None

    def __call__(self, x: Case) -> Self:
        return self.match(x)

    def __getitem__(self, attribute_name):
        return self.conditions.get(attribute_name, None)

    def match(self, x: Case) -> Rule:
        """
        Check if the rule or its refinement or its alternative matches the case.

        :param x: The case to match.
        :return: The rule that matches the case.
        """
        for att_name, condition in self.conditions.items():
            if att_name not in x.attributes:
                self.fired = False
                return self.alternative(x) if self.alternative else self
            elif not condition(x.attributes[att_name].value):
                self.fired = False
                return self.alternative(x) if self.alternative else self
        self.fired = True
        if self.refinement and self.refinement(x).fired:
            return self.refinement(x)
        else:
            return self

    def add_alternative(self, x: Case, conditions: Dict[str, Condition], category: Category):
        if self.alternative:
            self.alternative.add_alternative(x, conditions, category)
        else:
            self.alternative = Rule(conditions, category, corner_case=Case(x.id_, list(x.attributes.values())),
                                    parent=self, edge_weight="else if")

    def add_refinement(self, x: Case, conditions: Dict[str, Condition], category: Category):
        if self.refinement:
            self.refinement.add_alternative(x, conditions, category)
        else:
            self.refinement = Rule(conditions, category, corner_case=Case(x.id_, list(x.attributes.values())),
                                   parent=self, edge_weight="except if")

    def get_different_attributes(self, x: Case) -> Dict[str, Attribute]:
        return {a.name: a for a in self.corner_case.attributes.values()
                if a not in x.attributes.values()}

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def __str__(self, sep="\n"):
        conditions = f"^{sep}".join([str(c) for c in list(self.conditions.values())])
        conditions += f"{sep}=> {self.conclusion.name}"
        return conditions

    def __repr__(self):
        return self.__str__()


class SingleClassRDR:
    start_rule: Optional[Rule] = None

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

    def fit(self, x_batch: List[Case], y_batch: List[Category],
            n_iter: int = 100):
        plt.ion()
        # Start tree updating in a separate thread
        self.fig = plt.figure()
        all_pred = 0
        i = 0
        while all_pred != len(y_batch) and i < n_iter:
            all_pred = 0
            for x, y in zip(x_batch, y_batch):
                pred_cat = self.classify(x, y)
                all_pred += pred_cat == y
                self.draw_tree()
                i += 1
                if i >= n_iter:
                    break
        plt.ioff()
        print(f"Finished training in {i} iterations")
        plt.show()

    @staticmethod
    def edge_attr_debug(parent, child):
        if child is None:
            return ""
        weight = child.weight if child.weight is not None else 0  # Ensure weight is not None
        return f'style="bold", label=" {weight}"'  # Properly formatted

    def render_tree(self, use_dot_exporter: bool = False,
                    filename: str = "scrdr"):
        if not self.start_rule:
            logging.warning("No rules to render")
            return
        for pre, _, node in RenderTree(self.start_rule):
            print(f"{pre}{node.weight or ''} {node.__str__(sep='')}")
        if use_dot_exporter:
            de = DotExporter(self.start_rule,
                             edgeattrfunc=self.edge_attr_debug
                             )
            de.to_dotfile(f"{filename}{'.dot'}")
            de.to_picture(f"{filename}{'.png'}")

    def draw_tree(self):
        """Draw the tree using matplotlib and networkx."""
        if self.start_rule is None:
            return
        self.fig.clf()
        graph = tree_to_graph(self.start_rule)
        fig_sz_x = 10
        fig_sz_y = 9
        self.fig.set_size_inches(fig_sz_x, fig_sz_y)
        pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot")
        # scale down pos
        max_pos_x = max([v[0] for v in pos.values()])
        max_pos_y = max([v[1] for v in pos.values()])
        pos = {k: (v[0] * fig_sz_x / max_pos_x, v[1] * fig_sz_y / max_pos_y) for k, v in pos.items()}
        nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000,
                ax=self.fig.gca(), node_shape="o", font_size=8)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'),
                                     ax=self.fig.gca(), rotate=False, clip_on=False)
        plt.pause(0.1)
