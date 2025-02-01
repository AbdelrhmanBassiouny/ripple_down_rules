from __future__ import annotations

from sqlalchemy.util import OrderedSet
from typing_extensions import List, Any, Optional, Self, Dict, Callable

from episode_segmenter.utils import str_to_operator_fn


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


class Condition:
    def __init__(self, name: str, value: Any, func: Callable):
        self.name = name
        self.value = value
        self.func = func

    def __call__(self, x: Any) -> bool:
        return self.func(x, self.value)


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
        ljust = max(ljust, max([len(str(a.value)) for a in self.attributes.values()]))
        ljust += 2
        row1 = f"Case {self.id_} with attributes: \n"
        row2 = [f"{name.ljust(ljust)}" for name in names]
        row2 = "".join(row2) + "\n"
        row3 = [f"{str(self.attributes[name].value).ljust(ljust)}" for name in names]
        row3 = "".join(row3)
        return row1 + row2 + row3 + "\n"

    def __repr__(self):
        return str(self)


class Rule:
    category: Category
    fired: bool = False
    corner_case: Optional[Case] = None
    parent: Optional[Rule] = None
    refinement: Optional[Rule] = None
    alternative: Optional[Rule] = None

    def __init__(self, conditions: Dict[str, Condition], category: Category,
                 parent: Optional[Rule] = None,
                 corner_case: Optional[Case] = None):
        self.conditions = conditions
        self.corner_case = corner_case
        self.category = category
        self.parent = parent

    def __call__(self, x: Case) -> Self:
        return self.match(x)

    def __getitem__(self, attribute_name):
        return self.conditions.get(attribute_name, None)

    def match(self, x: Case) -> Rule:
        for att_name, condition in self.conditions.items():
            if att_name not in x.attributes:
                self.fired = False
                return self.alternative(x) if self.alternative else self
            elif not condition(x.attributes[att_name].value):
                self.fired = False
                return self.alternative(x) if self.alternative else self
        self.fired = True
        return self.refinement(x) if self.refinement else self

    def add_alternative(self, x: Case, conditions: Dict[str, Condition], category: Category):
        self.alternative = Rule(conditions, category, corner_case=Case(x.id_, list(x.attributes.values())),
                                parent=self)

    def add_refinement(self, x: Case, conditions: Dict[str, Condition], category: Category):
        self.refinement = Rule(conditions, category, corner_case=Case(x.id_, list(x.attributes.values())),
                               parent=self)

    def get_different_attributes(self, x: Case) -> Dict[str, Attribute]:
        return {a.name: a for a in self.corner_case.attributes if a not in x.attributes}


class SingleClassRDR:
    start_rule: Optional[Rule] = None

    def __init__(self, start_rule: Optional[Rule] = None):
        self.start_rule = start_rule

    def classify(self, x: Case, target: Optional[Category] = None) -> Category:
        if not self.start_rule:
            conditions = self.ask_user(x, target)
            self.start_rule = Rule(conditions, target, corner_case=Case(x.id_, list(x.attributes.values())))
        pred = self.start_rule(x)
        if target and pred.category != target:
            diff_attributes = pred.get_different_attributes(x)
            conditions = self.ask_user(x, target, pred, diff_attributes)
            if pred.fired:
                pred.add_refinement(x, conditions, target)
            else:
                pred.add_alternative(x, conditions, target)
        return pred.category

    @staticmethod
    def ask_user(x: Case, target: Category, pred: Optional[Rule] = None,
                 diff_attributes: Optional[Dict[str, Attribute]] = None) -> Dict[str, Condition]:
        if pred:
            action = "Refinement" if pred.fired else "Alternative"
            print(f"{action} needed for rule:\n")
            all_attributes = list(pred.corner_case.attributes.values()) + list(x.attributes.values())
        else:
            print("Please provide the first rule:")
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

        if pred:
            pred.corner_case.print_values(all_names, is_corner_case=True,
                                          ljust_sz=max_len)
        x.print_values(all_names, target, ljust_sz=max_len)

        if diff_attributes:
            diff = {name: "Y" if name in diff_attributes else "N" for name in all_names}
            print("".join([f" different : {ljust(diff[name])}" for name in all_names]))

        # take user input
        rule_conditions = {}
        print(f"Please provide the differentiating features using comparison operators: <, >, <=, >=, ==:")
        while True:
            value = input()
            rules = value.split(",")
            for rule in rules:
                rule = rule.strip()
                name, value, func = str_to_operator_fn(rule)
                if name and value and func:
                    if name not in all_names:
                        print(f"Attribute {name} not found in the attributes list please enter it again:")
                        break
                    rule_conditions[name] = Condition(name, int(value), func)
            return rule_conditions

    def fit(self, x_batch: List[Case], y_batch: List[Category]):
        all_pred = 0
        while all_pred != len(y_batch):
            all_pred = 0
            for x, y in zip(x_batch, y_batch):
                pred_cat = self.classify(x, y)
                all_pred += pred_cat == y
