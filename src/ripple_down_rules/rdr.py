from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy

from matplotlib import pyplot as plt
from ordered_set import OrderedSet
from sqlalchemy import Column
from sqlalchemy.orm import DeclarativeBase, Session, MappedColumn as Column
from typing_extensions import List, Optional, Dict, Type, Union, Any

from .datastructures import Condition, Case, MCRDRMode, Attribute, RDRMode
from .experts import Expert, Human
from .rules import Rule, SingleClassRule, MultiClassTopRule
from .utils import draw_tree, get_property_name


class RippleDownRules(ABC):
    """
    The abstract base class for the ripple down rules classifiers.
    """
    fig: Optional[plt.Figure] = None
    """
    The figure to draw the tree on.
    """
    expert_accepted_conclusions: Optional[List[Attribute]] = None
    """
    The conclusions that the expert has accepted, such that they are not asked again.
    """

    def __init__(self, start_rule: Optional[Rule] = None,
                 mode: RDRMode = RDRMode.Propositional):
        """
        :param start_rule: The starting rule for the classifier.
        :param mode: The mode of the classifier, either Propositional or Relational.
        """
        self.start_rule = start_rule
        self.fig: Optional[plt.Figure] = None
        self.mode: RDRMode = mode

    def __call__(self, x: Case) -> Attribute:
        return self.classify(x)

    @abstractmethod
    def classify(self, x: Case) -> Optional[Attribute]:
        """
        Classify a case.

        :param x: The case to classify.
        :return: The category that the case belongs to.
        """
        pass

    @abstractmethod
    def fit_case(self, x: Case, target: Optional[Attribute] = None,
                 expert: Optional[Expert] = None, **kwargs) -> Attribute:
        """
        Fit the RDR on a case, and ask the expert for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        pass

    def fit(self, x_batch: List[Case], y_batch: Optional[List[Attribute]] = None,
            expert: Optional[Expert] = None,
            n_iter: int = None,
            animate_tree: bool = False,
            **kwargs_for_fit_case):
        """
        Fit the classifier to a batch of cases and categories.

        :param x_batch: The batch of cases to fit the classifier to.
        :param y_batch: The batch of categories to fit the classifier to.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param n_iter: The number of iterations to fit the classifier for.
        :param animate_tree: Whether to draw the tree while fitting the classifier.
        :param kwargs_for_fit_case: The keyword arguments to pass to the fit_case method.
        """
        if animate_tree:
            plt.ion()
        i = 0
        stop_iterating = False
        num_rules: int = 0
        while not stop_iterating:
            all_pred = 0
            all_recall = []
            all_precision = []
            if not y_batch:
                y_batch = [None] * len(x_batch)
            for x, y in zip(x_batch, y_batch):
                if not y:
                    conclusions = self.classify(x) if self.start_rule and self.start_rule.conditions else []
                    y = expert.ask_for_conclusion(x, conclusions)
                pred_cat = self.fit_case(x, y, expert=expert, **kwargs_for_fit_case)
                pred_cat = pred_cat if isinstance(pred_cat, list) else [pred_cat]
                y = y if isinstance(y, list) else [y]
                recall = [not yi or (yi in pred_cat) for yi in y]
                y_type = [type(yi) for yi in y]
                precision = [(pred in y) or (type(pred) not in y_type) for pred in pred_cat]
                match = all(recall) and all(precision)
                all_recall.extend(recall)
                all_precision.extend(precision)
                if not match:
                    print(f"Predicted: {pred_cat} but expected: {y}")
                all_pred += int(match)
                if animate_tree and self.start_rule.size > num_rules:
                    num_rules = self.start_rule.size
                    self.update_figures()
                i += 1
                all_predicted = y_batch and all_pred == len(y_batch)
                num_iter_reached = n_iter and i >= n_iter
                stop_iterating = all_predicted or num_iter_reached
                if stop_iterating:
                    break
            print(f"Recall: {sum(all_recall) / len(all_recall)}")
            print(f"Precision: {sum(all_precision) / len(all_precision)}")
            print(f"Accuracy: {all_pred}/{n_iter}")
        print(f"Finished training in {i} iterations")
        if animate_tree:
            plt.ioff()
            plt.show()

    def update_figures(self):
        """
        Update the figures of the classifier.
        """
        if isinstance(self, GeneralRDR):
            for i, (_type, rdr) in enumerate(self.start_rules_dict.items()):
                if not rdr.fig:
                    rdr.fig = plt.figure(f"Rule {i}: {_type.__name__}")
                draw_tree(rdr.start_rule, rdr.fig)
        else:
            if not self.fig:
                self.fig = plt.figure(0)
            draw_tree(self.start_rule, self.fig)


class SingleClassRDR(RippleDownRules):
    table: Type[DeclarativeBase]
    target_column: Column

    def fit_case(self, case: Union[table, Case], target: Optional[Union[Attribute, Column]] = None,
                 expert: Optional[Expert] = None, for_attribute: Optional[Any] = None,
                 session: Optional[Session] = None,
                 **kwargs) -> Attribute:
        """
        Classify a case, and ask the user for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category if provided.

        :param case: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param for_attribute: The property of the case to find a value for.
        :param session: The SQLAlchemy session to use for the database.
        :return: The category that the case belongs to.
        """
        expert = expert if expert else Human(mode=self.mode)
        if not target:
            if isinstance(case, Case):
                for_attribute = case[case.get_property_name(for_attribute)]
            attribute_name = for_attribute.__class__.__name__
            target = expert.ask_for_conclusion(case, attribute_name=attribute_name, attribute_type=type(for_attribute),
                                               session=session)
        if not self.start_rule:
            conditions = expert.ask_for_conditions(case, [target], session=session)
            self.start_rule = SingleClassRule(conditions, target, corner_case=case)

        pred = self.evaluate(case)

        if pred.conclusion != target:
            conditions = expert.ask_for_conditions(case, [target], pred)
            pred.fit_rule(case, target, conditions=conditions)

        return self.classify(case)

    def classify(self, x: Case) -> Optional[Attribute]:
        """
        Classify a case by recursively evaluating the rules until a rule fires or the last rule is reached.
        """
        pred = self.evaluate(x)
        return pred.conclusion if pred.fired else None

    def evaluate(self, x: Case) -> SingleClassRule:
        """
        Evaluate the starting rule on a case.
        """
        matched_rule = self.start_rule(x)
        return matched_rule if matched_rule else self.start_rule


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
    conclusions: Optional[List[Attribute]] = None
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
        :param mode: The mode of the classifier, either StopOnly or StopPlusRule, or StopPlusRuleCombined.
        """
        self.start_rules = [MultiClassTopRule()] if not start_rules else start_rules
        super(MultiClassRDR, self).__init__(self.start_rules[0])
        self.mode: MCRDRMode = mode

    def classify(self, x: Case) -> List[Attribute]:
        evaluated_rule = self.start_rule
        self.conclusions = []
        while evaluated_rule:
            next_rule = evaluated_rule(x)
            if evaluated_rule.fired:
                self.add_conclusion(evaluated_rule)
            evaluated_rule = next_rule
        return self.conclusions

    def fit_case(self, x: Case, targets: Optional[Union[Attribute, List[Attribute]]] = None,
                 expert: Optional[Expert] = None, add_extra_conclusions: bool = False) -> List[Attribute]:
        """
        Classify a case, and ask the user for stopping rules or classifying rules if the classification is incorrect
         or missing by comparing the case with the target category if provided.

        :param x: The case to classify.
        :param targets: The target categories to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions or for extra conclusions.
        :param add_extra_conclusions: Whether to add extra conclusions after classification is done.
        :return: The conclusions that the case belongs to.
        """
        expert = expert if expert else Human()
        targets = targets if isinstance(targets, list) else [targets]
        self.expert_accepted_conclusions = []
        user_conclusions = []
        for target in targets:
            self.update_start_rule(x, target, expert)
            self.conclusions = []
            self.stop_rule_conditions = None
            evaluated_rule = self.start_rule
            while evaluated_rule:
                next_rule = evaluated_rule(x)
                good_conclusions = targets + user_conclusions + self.expert_accepted_conclusions

                if evaluated_rule.fired:
                    if target and evaluated_rule.conclusion not in good_conclusions:
                        if evaluated_rule.conclusion not in x:
                            # Rule fired and conclusion is different from target
                            self.stop_wrong_conclusion_else_add_it(x, target, expert, evaluated_rule,
                                                                   add_extra_conclusions)
                    else:
                        # Rule fired and target is correct or there is no target to compare
                        self.add_conclusion(evaluated_rule)

                if not next_rule:
                    if target not in self.conclusions:
                        # Nothing fired and there is a target that should have been in the conclusions
                        self.add_rule_for_case(x, target, expert)
                        # Have to check all rules again to make sure only this new rule fires
                        next_rule = self.start_rule
                    elif add_extra_conclusions and not user_conclusions:
                        # No more conclusions can be made, ask the expert for extra conclusions if needed.
                        user_conclusions.extend(self.ask_expert_for_extra_conclusions(expert, x))
                        if user_conclusions:
                            next_rule = self.last_top_rule
                evaluated_rule = next_rule
        return list(OrderedSet(self.conclusions))

    def update_start_rule(self, x: Case, target: Attribute, expert: Expert):
        """
        Update the starting rule of the classifier.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if not self.start_rule.conditions:
            conditions = expert.ask_for_conditions(x, target)
            self.start_rule.conditions = conditions
            self.start_rule.conclusion = target
            self.start_rule.corner_case = Case(x._id, x._attributes_list)

    @property
    def last_top_rule(self) -> Optional[MultiClassTopRule]:
        """
        Get the last top rule in the tree.
        """
        if not self.start_rule.furthest_alternative:
            return self.start_rule
        else:
            return self.start_rule.furthest_alternative[-1]

    def stop_wrong_conclusion_else_add_it(self, x: Case, target: Attribute, expert: Expert,
                                          evaluated_rule: MultiClassTopRule,
                                          add_extra_conclusions: bool):
        """
        Stop a wrong conclusion by adding a stopping rule.
        """
        if self.is_same_category_type(evaluated_rule.conclusion, target) \
                and self.is_conflicting_with_target(evaluated_rule.conclusion, target):
            self.stop_conclusion(x, target, expert, evaluated_rule)
        elif not self.conclusion_is_correct(x, target, expert, evaluated_rule, add_extra_conclusions):
            self.stop_conclusion(x, target, expert, evaluated_rule)

    def stop_conclusion(self, x: Case, target: Attribute, expert: Expert, evaluated_rule: MultiClassTopRule):
        """
        Stop a conclusion by adding a stopping rule.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param evaluated_rule: The evaluated rule to ask the expert about.
        """
        conditions = expert.ask_for_conditions(x, target, evaluated_rule)
        evaluated_rule.fit_rule(x, target, conditions=conditions)
        if self.mode == MCRDRMode.StopPlusRule:
            self.stop_rule_conditions = conditions
        if self.mode == MCRDRMode.StopPlusRuleCombined:
            new_top_rule_conditions = {**evaluated_rule.conditions, **conditions}
            self.add_top_rule(new_top_rule_conditions, target, x)

    @staticmethod
    def is_conflicting_with_target(conclusion: Attribute, target: Attribute) -> bool:
        """
        Check if the conclusion is conflicting with the target category.

        :param conclusion: The conclusion to check.
        :param target: The target category to compare the conclusion with.
        :return: Whether the conclusion is conflicting with the target category.
        """
        if conclusion._mutually_exclusive:
            return True
        else:
            return not conclusion._value.issubset(target._value)

    @staticmethod
    def is_same_category_type(conclusion: Attribute, target: Attribute) -> bool:
        """
        Check if the conclusion is of the same class as the target category.

        :param conclusion: The conclusion to check.
        :param target: The target category to compare the conclusion with.
        :return: Whether the conclusion is of the same class as the target category but has a different value.
        """
        return conclusion.__class__ == target.__class__ and target.__class__ != Attribute

    def conclusion_is_correct(self, x: Case, target: Attribute, expert: Expert, evaluated_rule: Rule,
                              add_extra_conclusions: bool) -> bool:
        """
        Ask the expert if the conclusion is correct, and add it to the conclusions if it is.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param evaluated_rule: The evaluated rule to ask the expert about.
        :param add_extra_conclusions: Whether adding extra conclusions after classification is allowed.
        :return: Whether the conclusion is correct or not.
        """
        conclusions = list(OrderedSet(self.conclusions))
        if (add_extra_conclusions and expert.ask_if_conclusion_is_correct(x, evaluated_rule.conclusion,
                                                                          targets=target,
                                                                          current_conclusions=conclusions)):
            self.add_conclusion(evaluated_rule)
            self.expert_accepted_conclusions.append(evaluated_rule.conclusion)
            return True
        return False

    def add_rule_for_case(self, x: Case, target: Attribute, expert: Expert):
        """
        Add a rule for a case that has not been classified with any conclusion.
        """
        if self.stop_rule_conditions and self.mode == MCRDRMode.StopPlusRule:
            conditions = self.stop_rule_conditions
            self.stop_rule_conditions = None
        else:
            conditions = expert.ask_for_conditions(x, target)
        self.add_top_rule(conditions, target, x)

    def ask_expert_for_extra_conclusions(self, expert: Expert, x: Case) -> List[Attribute]:
        """
        Ask the expert for extra conclusions when no more conclusions can be made.

        :param expert: The expert to ask for extra conclusions.
        :param x: The case to ask extra conclusions for.
        :return: The extra conclusions that the expert has provided.
        """
        extra_conclusions = []
        conclusions = list(OrderedSet(self.conclusions))
        if not expert.use_loaded_answers:
            print("current conclusions:", conclusions)
        extra_conclusions_dict = expert.ask_for_extra_conclusions(x, conclusions)
        if extra_conclusions_dict:
            for conclusion, conditions in extra_conclusions_dict.items():
                self.add_top_rule(conditions, conclusion, x)
                extra_conclusions.append(conclusion)
        return extra_conclusions

    def add_conclusion(self, evaluated_rule: Rule) -> None:
        """
        Add the conclusion of the evaluated rule to the list of conclusions.

        :param evaluated_rule: The evaluated rule to add the conclusion of.
        """
        conclusion_types = [type(c) for c in self.conclusions]
        if type(evaluated_rule.conclusion) not in conclusion_types:
            self.conclusions.append(evaluated_rule.conclusion)
        else:
            same_type_conclusions = [c for c in self.conclusions if type(c) == type(evaluated_rule.conclusion)]
            combined_conclusion = evaluated_rule.conclusion._value if isinstance(evaluated_rule.conclusion._value, set) \
                else {evaluated_rule.conclusion._value}
            category_type = type(evaluated_rule.conclusion)
            for c in same_type_conclusions:
                combined_conclusion.update(c._value if isinstance(c._value, set) else {c._value})
                self.conclusions.remove(c)
            self.conclusions.append(category_type(combined_conclusion))

    def add_top_rule(self, conditions: Dict[str, Condition], conclusion: Attribute, corner_case: Case):
        """
        Add a top rule to the classifier, which is a rule that is always checked and is part of the start_rules list.

        :param conditions: The conditions of the rule.
        :param conclusion: The conclusion of the rule.
        :param corner_case: The corner case of the rule.
        """
        self.start_rule.alternative = MultiClassTopRule(conditions, conclusion, corner_case=corner_case)


class GeneralRDR(RippleDownRules):
    """
    A general ripple down rules classifier, which can draw multiple conclusions for a case, but each conclusion is part
    of a set of mutually exclusive conclusions. Whenever a conclusion is made, the classification restarts from the
    starting rule, and all the rules that belong to the class of the made conclusion are not checked again. This
    continues until no more rules can be fired. In addition, previous conclusions can be used as conditions or input to
    the next classification/cycle.
    Another possible mode is to have rules that are considered final, when fired, inference will not be restarted,
     and only a refinement can be made to the final rule, those can also be used in another SCRDR of their own that
     gets called when the final rule fires.
    """

    def __init__(self, category_rdr_map: Optional[Dict[Type[Attribute], Union[SingleClassRDR, MultiClassRDR]]] = None):
        """
        :param category_rdr_map: A map of categories to ripple down rules classifiers,
        where each category is a parent category that has a set of mutually exclusive (in case of SCRDR) child
        categories, e.g. {Species: SCRDR, Habitat: MCRDR}, where Species and Habitat are parent categories and SCRDR
        and MCRDR are SingleClass and MultiClass ripple down rules classifiers. Species can have child categories like
        Mammal, Bird, Fish, etc. which are mutually exclusive, and Habitat can have child categories like
        Land, Water, Air, etc, which are not mutually exclusive due to some animals living more than one habitat.
        """
        self.start_rules_dict: Dict[Type[Attribute], Union[SingleClassRDR, MultiClassRDR]] \
            = category_rdr_map if category_rdr_map else {}
        super(GeneralRDR, self).__init__()
        self.all_figs: List[plt.Figure] = [sr.fig for sr in self.start_rules_dict.values()]

    @property
    def start_rule(self) -> Optional[Union[SingleClassRule, MultiClassTopRule]]:
        return self.start_rules[0] if self.start_rules_dict else None

    @start_rule.setter
    def start_rule(self, value: Union[SingleClassRDR, MultiClassRDR]):
        if value:
            self.start_rules_dict[type(value.start_rule.conclusion)] = value

    @property
    def start_rules(self) -> List[Union[SingleClassRule, MultiClassTopRule]]:
        return [rdr.start_rule for rdr in self.start_rules_dict.values()]

    def classify(self, x: Case) -> Optional[List[Attribute]]:
        """
        Classify a case by going through all RDRs and adding the categories that are classified, and then restarting
        the classification until no more categories can be added.

        :param x: The case to classify.
        :return: The categories that the case belongs to.
        """
        conclusions = []
        x_cp = copy(x)
        while True:
            added_attributes = False
            for cat_type, rdr in self.start_rules_dict.items():
                if cat_type in x_cp:
                    continue
                pred_atts = rdr.classify(x_cp)
                if pred_atts:
                    pred_atts = pred_atts if isinstance(pred_atts, list) else [pred_atts]
                    added_attributes = True
                    for pred_att in pred_atts:
                        x_cp.add_attribute(pred_att)
                        conclusions.append(pred_att)
            if not added_attributes:
                break
        return list(OrderedSet(conclusions))

    def fit_case(self, x: Case, targets: Optional[Union[Attribute, List[Attribute]]] = None,
                 expert: Optional[Expert] = None,
                 **kwargs) -> List[Attribute]:
        """
        Fit the GRDR on a case, if the target is a new type of category, a new RDR is created for it,
        else the existing RDR of that type will be fitted on the case, and then classification is done and all
        concluded categories are returned. If the category is mutually exclusive, an SCRDR is created, else an MCRDR.
        In case of SCRDR, multiple conclusions of the same type replace each other, in case of MCRDR, they are added if
        they are accepted by the expert, and the attribute of that category is represented in the case as a set of
        values.
        """
        expert = expert if expert else Human()
        if not targets:
            return self.classify(x)
        targets = targets if isinstance(targets, list) else [targets]
        for t in targets:
            x_cp = copy(x)
            if type(t) not in self.start_rules_dict:
                conclusions = self.classify(x)
                x_cp.add_attributes(conclusions)
                new_rdr = SingleClassRDR() if t._mutually_exclusive else MultiClassRDR()
                new_conclusions = new_rdr.fit_case(x_cp, t, expert, **kwargs)
                self.start_rules_dict[type(t)] = new_rdr
                x_cp.add_attributes(new_conclusions)
            elif type(t) not in x_cp:
                for rdr_type, rdr in self.start_rules_dict.items():
                    if type(t) != rdr_type:
                        conclusions = rdr.classify(x_cp)
                    else:
                        conclusions = self.start_rules_dict[type(t)].fit_case(x_cp, t, expert, **kwargs)
                    x_cp.add_attributes(conclusions)

        return self.classify(x)

    @property
    def names_of_all_types(self) -> List[str]:
        """
        Get the names of all the types of categories that the GRDR can classify.
        """
        return [t.__name__ for t in self.start_rules_dict.keys()]

    @property
    def all_types(self) -> List[Type[Attribute]]:
        """
        Get all the types of categories that the GRDR can classify.
        """
        return list(self.start_rules_dict.keys())
