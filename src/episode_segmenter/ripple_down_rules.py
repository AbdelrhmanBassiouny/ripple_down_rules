from __future__ import annotations

import time
from abc import ABC, abstractmethod

from typing_extensions import List, Any, Optional


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
    corner_case: Optional[Case] = None
    refinement: Optional[Rule] = None
    alternative: Optional[Rule] = None

    def __init__(self, attributes: List[Attribute], category: Category,
                 corner_case: Optional[Case] = None):
        self.attributes = attributes
        self.corner_case = corner_case
        self.category = category

    def __call__(self, x: Case):
        return self.match(x)

    def match(self, x: Case) -> Category:
        for attribute in self.attributes:
            if attribute not in x.attributes:
                return self.alternative(x) if self.alternative else None
        return self.refinement(x) if self.refinement else self.category


class SingleClassRDR:
    categories: List[Category]
    start_rule: Rule

    def __init__(self, categories: List[Category], start_rule: Rule):
        self.categories = categories
        self.start_rule = start_rule

    def classify(self, x: Case) -> Category:
        return self.start_rule(x)
