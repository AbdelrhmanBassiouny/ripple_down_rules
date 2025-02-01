from __future__ import annotations

import time
from abc import ABC, abstractmethod

from typing_extensions import List, Any, Optional, Self


class Category:
    def __init__(self, name: str):
        self.name = name


class Attribute:
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value


class Case:
    def __init__(self, attributes: List[Attribute]):
        self.attributes = attributes


class Rule:

    category: Category
    value: bool = False
    corner_case: Optional[Case] = None
    refinement: Optional[Rule] = None
    alternative: Optional[Rule] = None

    def __init__(self, attributes: List[Attribute], category: Category,
                 corner_case: Optional[Case] = None):
        self.attributes = attributes
        self.corner_case = corner_case
        self.category = category

    def __call__(self, x: Case) -> Self:
        return self.match(x)

    def match(self, x: Case) -> Rule:
        for attribute in self.attributes:
            if attribute not in x.attributes:
                self.value = False
                return self.alternative(x) if self.alternative else self
        self.value = True
        return self.refinement(x) if self.refinement else self


class SingleClassRDR:
    categories: List[Category]
    start_rule: Rule

    def __init__(self, categories: List[Category], start_rule: Rule):
        self.categories = categories
        self.start_rule = start_rule

    def classify(self, x: Case, target: Optional[Category] = None) -> Category:
        pred = self.start_rule(x)
        if target and pred.category != target:
            self.ask(x, pred)
        return pred.category

    def ask(self, x: Case, pred: Rule):
        # TODO: This is a placeholder for a real ask function
        pred.refinement = Rule(x.attributes, pred.category)

