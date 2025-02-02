from __future__ import annotations

import logging

import networkx as nx
from anytree import RenderTree, NodeMixin
from anytree.exporter import DotExporter
from matplotlib import pyplot as plt
from sqlalchemy.util import OrderedSet
from typing_extensions import List, Any, Optional, Self, Dict, Callable, Tuple

from pyrdr.utils import tree_to_graph


class Category:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name


class Attribute:
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def __hash__(self):
        return hash((self.name, self.value))


class Operator:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func

    def __call__(self, x: Any, y: Any) -> bool:
        return self.func(x, y)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class Equal(Operator):
    def __init__(self):
        super().__init__("==", lambda x, y: x == y)


class Greater(Operator):
    def __init__(self):
        super().__init__(">", lambda x, y: x > y)


class GreaterEqual(Operator):
    def __init__(self):
        super().__init__(">=", lambda x, y: x >= y)


class Less(Operator):
    def __init__(self):
        super().__init__("<", lambda x, y: x < y)


class LessEqual(Operator):

    def __init__(self):
        super().__init__("<=", lambda x, y: x <= y)


def str_to_operator_fn(rule_str: str) -> Tuple[Optional[str], Optional[str], Optional[Callable]]:
    """
    Convert a string containing a rule to a function that represents the rule.

    :param rule_str: A string that contains the rule.
    :return: An operator object and two arguments that represents the rule.
    """
    operator: Optional[Operator] = None
    arg1: Optional[str] = None
    arg2: Optional[str] = None
    if LessEqual().__str__() in rule_str:
        operator = LessEqual()
    elif GreaterEqual().__str__() in rule_str:
        operator = GreaterEqual()
    elif Equal().__str__() in rule_str:
        operator = Equal()
    elif Less().__str__() in rule_str:
        operator = Less()
    elif Greater().__str__() in rule_str:
        operator = Greater()
    if operator is not None:
        arg1, arg2 = rule_str.split(operator.__str__())
        arg1 = arg1.strip()
        arg2 = arg2.strip()
    return arg1, arg2, operator


class Condition:
    def __init__(self, name: str, value: Any, operator: Operator):
        self.name = name
        self.value = value
        self.operator = operator

    def __call__(self, x: Any) -> bool:
        return self.operator(x, self.value)

    def __str__(self):
        return f"{self.name} {self.operator} {self.value}"

    def __repr__(self):
        return self.__str__()


class Case:
    def __init__(self, id_: str, attributes: List[Attribute]):
        self.attributes = {a.name: a for a in attributes}
        self.id_ = id_

    @property
    def attribute_values(self):
        return [a.value for a in self.attributes.values()]

    def __eq__(self, other):
        return self.attributes == other.attributes

    def __getitem__(self, attribute_name):
        return self.attributes.get(attribute_name, None)

    def ljust(self, s, sz=15):
        return str(s).ljust(sz)

    def print_values(self, all_names: Optional[List[str]] = None,
                     target: Optional[Category] = None,
                     is_corner_case: bool = False,
                     ljust_sz: int = 15):
        all_names = list(self.attributes.keys()) if not all_names else all_names
        if is_corner_case:
            case_row = self.ljust(f"corner case: ", sz=ljust_sz)
        else:
            case_row = self.ljust(f"case: ", sz=ljust_sz)
        case_row += self.ljust(self.id_, sz=ljust_sz)
        case_row += "".join([f"{self.ljust(self[name].value, sz=ljust_sz)}"
                             for name in all_names])
        if target:
            case_row += f"{self.ljust(target.name, sz=ljust_sz)}"
        print(case_row)

    def __str__(self):
        names = list(self.attributes.keys())
        ljust = max([len(name) for name in names])
        ljust = max(ljust, max([len(str(a.value)) for a in self.attributes.values()])) + 2
        row1 = f"Case {self.id_} with attributes: \n"
        row2 = [f"{name.ljust(ljust)}" for name in names]
        row2 = "".join(row2) + "\n"
        row3 = [f"{str(self.attributes[name].value).ljust(ljust)}" for name in names]
        row3 = "".join(row3)
        return row1 + row2 + row3 + "\n"

    def __repr__(self):
        return self.__str__()


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

    def classify(self, x: Case, target: Optional[Category] = None) -> Category:
        """
        Classify a case, and ask the user for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category if provided.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :return: The category that the case belongs to.
        """
        if not self.start_rule:
            conditions = self.ask_user(x, target)
            self.start_rule = Rule(conditions, target, corner_case=Case(x.id_, list(x.attributes.values())))

        pred = self.start_rule(x)

        if target and pred.conclusion != target:
            diff_attributes = pred.get_different_attributes(x)
            conditions = self.ask_user(x, target, pred, diff_attributes)
            if pred.fired:
                pred.add_refinement(x, conditions, target)
            else:
                pred.add_alternative(x, conditions, target)

        return pred.conclusion

    @staticmethod
    def ask_user(x: Case, target: Category, pred: Optional[Rule] = None,
                 diff_attributes: Optional[Dict[str, Attribute]] = None) -> Dict[str, Condition]:
        if pred:
            action = "Refinement" if pred.fired else "Alternative"
            print(f"{action} needed for rule:\n")
        if pred and pred.fired:
            all_attributes = list(pred.corner_case.attributes.values()) + list(x.attributes.values())
        else:
            print("Please provide a rule for case:")
            all_attributes = list(x.attributes.values())

        all_names = OrderedSet([a.name for a in all_attributes])
        max_len = max([len(name) for name in all_names])
        max_len = max(max_len, max([len(str(a.value)) for a in all_attributes])) + 4

        def ljust(s):
            return str(s).ljust(max_len)

        names_row = ljust(f"names: ")
        names_row += ljust("id")
        names_row += "".join([f"{ljust(name)}" for name in all_names + ["type"]])
        print(names_row)

        if pred and pred.fired:
            pred.corner_case.print_values(all_names, is_corner_case=True,
                                          ljust_sz=max_len)
        x.print_values(all_names, target, ljust_sz=max_len)

        if pred and pred.fired and diff_attributes:
            diff = {name: "Y" if name in diff_attributes else "N" for name in all_names}
            diff_row = ljust(f"diff: ") + ljust(" ")
            diff_row += "".join([f"{ljust(diff[name])}" for name in all_names])
            print(diff_row)

        # take user input
        rule_conditions = {}
        print(f"Please provide the differentiating features using comparison operators: <, >, <=, >=, ==:")
        while True:
            value = input()
            rules = value.split(",")
            done = True
            messages = []
            for rule in rules:
                rule = rule.strip()
                name, value, operator = str_to_operator_fn(rule)
                if name and value and operator:
                    if name not in all_names:
                        messages.append(f"Attribute {name} not found in the attributes list please enter it again")
                        done = False
                        continue
                    rule_conditions[name] = Condition(name, int(value), operator)
            if done:
                return rule_conditions
            elif len(messages) > 0:
                print("\n".join(messages))

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
