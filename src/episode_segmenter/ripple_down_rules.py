from __future__ import annotations

from typing_extensions import List, Any, Optional, Self, Dict


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


class Case:
    def __init__(self, attributes: List[Attribute]):
        self.attributes = {a.name: a for a in attributes}

    @property
    def attribute_values(self):
        return [a.value for a in self.attributes.values()]

    def __eq__(self, other):
        return self.attributes == other.attributes

    def __getitem__(self, attribute_name):
        return self.attributes.get(attribute_name, None)

    def __str__(self):
        names = list(self.attributes.keys())
        ljust = max([len(name) for name in names])
        ljust = max(ljust, max([len(str(a.value)) for a in self.attributes.values()]))
        ljust += 2
        row1 = "Case with attributes: \n"
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

    def __init__(self, attributes: List[Attribute], category: Category,
                 parent: Optional[Rule] = None,
                 corner_case: Optional[Case] = None):
        self.attributes = attributes
        self.corner_case = corner_case
        self.category = category
        self.parent = parent

    def __call__(self, x: Case) -> Self:
        return self.match(x)

    def match(self, x: Case) -> Rule:
        for attribute in self.attributes:
            if attribute not in x.attributes:
                self.fired = False
                return self.alternative(x) if self.alternative else self
        self.fired = True
        return self.refinement(x) if self.refinement else self

    def add_alternative(self, x: Case, attributes: List[Attribute]):
        self.alternative = Rule(attributes, self.category, corner_case=Case(x.attributes),
                                parent=self)

    def add_refinement(self, x: Case, attributes: List[Attribute]):
        self.refinement = Rule(attributes, self.category, corner_case=Case(x.attributes),
                               parent=self)

    def get_different_attributes(self, x: Case) -> Dict[str, Attribute]:
        return {a.name: a for a in self.corner_case.attributes if a not in x.attributes}


class SingleClassRDR:
    start_rule: Rule

    def __init__(self, start_rule: Rule):
        self.start_rule = start_rule

    def classify(self, x: Case, target: Optional[Category] = None) -> Category:
        pred = self.start_rule(x)
        if target:
            diff_attributes = pred.get_different_attributes(x)
            self.ask_user(pred, x, diff_attributes)
            if pred.fired and pred.category != target:
                pred.add_refinement(x, diff_attributes.values())
            elif not pred.fired:
                pred.add_alternative(x, diff_attributes.values())
        return pred.category

    @staticmethod
    def ask_user(pred: Rule, x: Case, diff_attributes: Dict[str, Attribute]):
        action = "Refinement" if pred.fired else "Alternative"
        print(f"{action} needed for rule:\n")
        all_names = set([a.name for a in pred.corner_case.attributes + x.attributes])
        max_len = max([len(name) for name in all_names])
        max_len = max(max_len, max([len(str(a.value)) for a in pred.corner_case.attributes + x.attributes]))
        diff = {name: "Y" if name in diff_attributes else "N" for name in all_names}

        def ljust(s): return s.ljust(max_len)

        print(f"             {ljust(a.name)}" for a in all_names)
        print(f"corner case: {ljust(pred.corner_case[name])}" for name in all_names)
        print(f"    case   : {ljust(x[name])}" for name in all_names)
        print(f" different : {ljust(diff[name])}" for name in all_names)

    def fit(self, x_batch: List[Case], y_batch: List[Category]):
        all_pred = 0
        while all_pred != len(y_batch):
            all_pred = 0
            for x, y in zip(x_batch, y_batch):
                pred_cat = self.classify(x, y)
                all_pred += pred_cat == y
