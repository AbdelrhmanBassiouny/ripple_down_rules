from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy

from matplotlib import pyplot as plt
from orderedset import OrderedSet
from typing_extensions import List, Optional, Dict, Type, Union

from .datastructures import Category, Condition, Case, Stop, MCRDRMode, Attribute
from .experts import Expert, Human
from .rules import Rule, SingleClassRule, MultiClassTopRule, MultiClassStopRule
from .utils import draw_tree


class RippleDownRules(ABC):
    """
    The abstract base class for the ripple down rules classifiers.
    """
    fig: Optional[plt.Figure] = None
    """
    The figure to draw the tree on.
    """
    expert_accepted_conclusions: Optional[List[Category]] = None
    """
    The conclusions that the expert has accepted, such that they are not asked again.
    """

    def __init__(self, start_rule: Optional[Rule] = None):
        """
        :param start_rule: The starting rule for the classifier.
        """
        self.start_rule = start_rule
        self.fig: Optional[plt.Figure] = None

    def __call__(self, x: Case) -> Category:
        return self.classify(x)

    @abstractmethod
    def classify(self, x: Case) -> Optional[Category]:
        """
        Classify a case.

        :param x: The case to classify.
        :return: The category that the case belongs to.
        """
        pass

    @abstractmethod
    def fit_case(self, x: Case, target: Category,
                 expert: Optional[Expert] = None, **kwargs) -> Category:
        """
        Fit the RDR on a case, and ask the expert for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category.

        :param x: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        pass

    def fit(self, x_batch: List[Case], y_batch: List[Category],
            expert: Optional[Expert] = None,
            n_iter: int = None,
            animate_tree: bool = False,
            **kwargs_for_classify):
        """
        Fit the classifier to a batch of cases and categories.

        :param x_batch: The batch of cases to fit the classifier to.
        :param y_batch: The batch of categories to fit the classifier to.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param n_iter: The number of iterations to fit the classifier for.
        :param animate_tree: Whether to draw the tree while fitting the classifier.
        """
        if animate_tree:
            plt.ion()
            self.fig = plt.figure()
        all_pred = 0
        i = 0
        while (all_pred != len(y_batch) and n_iter and i < n_iter) \
                or (not n_iter and all_pred != len(y_batch)):
            all_pred = 0
            all_recall = []
            all_precision = []
            for x, y in zip(x_batch, y_batch):
                pred_cat = self.fit_case(x, y, expert=expert, **kwargs_for_classify)
                pred_cat = pred_cat if isinstance(pred_cat, list) else [pred_cat]
                y = y if isinstance(y, list) else [y]
                recall = [yi in pred_cat for yi in y]
                y_type = [type(yi) for yi in y]
                precision = [(pred in y) or (type(pred) not in y_type) for pred in pred_cat]
                match = all(recall) and all(precision)
                all_recall.extend(recall)
                all_precision.extend(precision)
                if not match:
                    print(f"Predicted: {pred_cat} but expected: {y}")
                all_pred += int(match)
                if animate_tree:
                    draw_tree(self.start_rule, self.fig)
                i += 1
                if n_iter and i >= n_iter:
                    break
            print(f"Recall: {sum(all_recall) / len(all_recall)}")
            print(f"Precision: {sum(all_precision) / len(all_precision)}")
            print(f"Accuracy: {all_pred}/{len(y_batch)}")
        print(f"Finished training in {i} iterations")
        if animate_tree:
            plt.ioff()
            plt.show()


class SingleClassRDR(RippleDownRules):

    def fit_case(self, x: Case, target: Category,
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
            self.start_rule = SingleClassRule(conditions, target, corner_case=Case(x.id_, x.attributes_list))

        pred = self.evaluate(x)

        if pred.conclusion != target:
            conditions = expert.ask_for_conditions(x, target, pred)
            pred.fit_rule(x, target, conditions=conditions)

        return self.classify(x)

    def classify(self, x: Case) -> Optional[Category]:
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
        :param mode: The mode of the classifier, either StopOnly or StopPlusRule, or StopPlusRuleCombined.
        """
        self.start_rules = [MultiClassTopRule()] if not start_rules else start_rules
        super(MultiClassRDR, self).__init__(self.start_rules[0])
        self.mode: MCRDRMode = mode

    def classify(self, x: Case) -> List[Category]:
        evaluated_rule = self.start_rule
        self.conclusions = []
        while evaluated_rule:
            next_rule = evaluated_rule(x)
            if evaluated_rule.fired:
                self.add_conclusion(evaluated_rule)
            evaluated_rule = next_rule
        return self.conclusions

    def fit_case(self, x: Case, targets: Union[Category, List[Category]],
                 expert: Optional[Expert] = None, add_extra_conclusions: bool = False) -> List[Category]:
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

                if evaluated_rule.fired and evaluated_rule.conclusion != Stop():
                    if target and evaluated_rule.conclusion not in [*targets, *user_conclusions] \
                            and Attribute.from_category(evaluated_rule.conclusion) not in x:
                        # Rule fired and conclusion is different from target
                        self.stop_wrong_conclusion_else_add_it(x, target, expert, evaluated_rule, add_extra_conclusions)
                    else:
                        # Rule fired and target is correct or there is no target to compare
                        self.add_conclusion(evaluated_rule)

                if target and not next_rule:
                    if target not in self.conclusions:
                        # Nothing fired and there is a target that should have been in the conclusions
                        self.add_rule_for_case(x, target, expert)
                        # Have to check all rules again to make sure only this new rule fires
                        evaluated_rule = self.start_rule
                        continue
                    elif add_extra_conclusions and not user_conclusions:
                        # No more conclusions can be made, ask the expert for extra conclusions if needed.
                        user_conclusions.extend(self.ask_expert_for_extra_conclusions(expert, x))
                        if user_conclusions:
                            evaluated_rule = self.last_top_rule
                            continue
                evaluated_rule = next_rule
        return list(OrderedSet(self.conclusions))

    def update_start_rule(self, x: Case, target: Category, expert: Expert):
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
            self.start_rule.corner_case = Case(x.id_, x.attributes_list)

    @property
    def last_top_rule(self) -> Optional[MultiClassTopRule]:
        """
        Get the last top rule in the tree.
        """
        if not self.start_rule.furthest_alternative:
            return self.start_rule
        else:
            return self.start_rule.furthest_alternative[-1]

    def is_last_rule(self, rule_idx: int) -> bool:
        """
        Check if the rule index is the last rule in the classifier.

        :param rule_idx: The index of the rule to check.
        :return: Whether the rule index is the last rule in the classifier.
        """
        return rule_idx == len(self.start_rules) - 1

    def stop_wrong_conclusion_else_add_it(self, x: Case, target: Category, expert: Expert,
                                          evaluated_rule: MultiClassTopRule,
                                          add_extra_conclusions: bool):
        """
        Stop a wrong conclusion by adding a stopping rule.
        """
        if evaluated_rule.conclusion in self.expert_accepted_conclusions:
            return
        elif not self.conclusion_is_correct(x, target, expert, evaluated_rule, add_extra_conclusions):
            conditions = expert.ask_for_conditions(x, target, evaluated_rule)
            evaluated_rule.fit_rule(x, target, conditions=conditions)
            if self.mode == MCRDRMode.StopPlusRule:
                self.stop_rule_conditions = conditions
            if self.mode == MCRDRMode.StopPlusRuleCombined:
                new_top_rule_conditions = {**evaluated_rule.conditions, **conditions}
                self.add_top_rule(new_top_rule_conditions, target, x)

    def conclusion_is_correct(self, x: Case, target: Category, expert: Expert, evaluated_rule: Rule,
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

    def add_rule_for_case(self, x: Case, target: Category, expert: Expert):
        """
        Add a rule for a case that has not been classified with any conclusion.
        """
        if self.stop_rule_conditions and self.mode == MCRDRMode.StopPlusRule:
            conditions = self.stop_rule_conditions
            self.stop_rule_conditions = None
        else:
            conditions = expert.ask_for_conditions(x, target)
        self.add_top_rule(conditions, target, x)

    def ask_expert_for_extra_conclusions(self, expert: Expert, x: Case) -> List[Category]:
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

    def add_conclusion(self, evaluated_rule: Rule):
        """
        Add the conclusion of the evaluated rule to the list of conclusions.
        """
        self.conclusions.append(evaluated_rule.conclusion)

    def add_top_rule(self, conditions: Dict[str, Condition], conclusion: Category, corner_case: Case):
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

    def __init__(self, category_scrdr_map: Optional[Dict[Type[Category], Union[SingleClassRDR, MultiClassRDR]]] = None):
        """
        :param category_scrdr_map: A map of categories to single class ripple down rules classifiers,
        where each category is a parent category that has a set of mutually exclusive child categories,
        e.g. {Species: SCRDR1, Habitat: SCRDR2}, where Species and Habitat are parent categories and SCRDR1 and SCRDR2
        are single class ripple down rules classifiers. Species can have child categories like Mammal, Bird, Fish, etc.
         which are mutually exclusive, and Habitat can have child categories like Land, Water, Air, etc, which are also
            mutually exclusive.
        """
        self.start_rules_dict: Dict[Type[Category], Union[SingleClassRDR, MultiClassRDR]] \
            = category_scrdr_map if category_scrdr_map else {}
        super(GeneralRDR, self).__init__()

    @property
    def start_rule(self) -> Optional[Union[SingleClassRule, MultiClassTopRule]]:
        return list(self.start_rules_dict.values())[0].start_rule if self.start_rules_dict else None

    @start_rule.setter
    def start_rule(self, value: Union[SingleClassRDR, MultiClassRDR]):
        if value:
            self.start_rules_dict[type(value.start_rule.conclusion)] = value

    def classify(self, x: Case) -> Optional[List[Category]]:
        """
        Classify a case by going through all SCRDRs and adding the categories that are classified, and then restarting
        the classification until no more categories can be added.

        :param x: The case to classify.
        :return: The categories that the case belongs to.
        """
        conclusions = []
        x_cp = copy(x)
        while True:
            added_attributes = False
            for cat_type, scrdr in self.start_rules_dict.items():
                if cat_type in x_cp:
                    continue
                pred_cats = scrdr.classify(x_cp)
                pred_cats = pred_cats if isinstance(pred_cats, list) else [pred_cats]
                if pred_cats:
                    added_attributes = True
                    for pred_cat in pred_cats:
                        x_cp.add_attribute_from_category(pred_cat)
                        conclusions.append(pred_cat)
            if not added_attributes:
                break
        return conclusions

    def fit_case(self, x: Case, targets: Union[Category, List[Category]],
                 expert: Optional[Expert] = None,
                 **kwargs) -> List[Category]:
        """
        Fit the GRDR on a case, if the target is a new type of category, a new SCRDR is created for it,
        else the existing SCRDR of that type will be fitted on the case, and then classification is done and all
        concluded categories are returned.
        """
        if not isinstance(targets, list):
            targets = [targets]
        conclusions = self.classify(x)
        x_cp = copy(x)
        x_cp.add_attributes_from_categories(conclusions)
        till_now_targets: List[Category] = []
        for t in targets:
            till_now_targets.append(t)
            new_conclusions: Optional[List[Category]] = None
            if type(t) not in self.start_rules_dict:
                new_rdr = SingleClassRDR() if type(t).mutually_exclusive else MultiClassRDR()
                self.start_rules_dict[type(t)] = new_rdr
                new_conclusions = new_rdr.fit_case(x_cp, t, expert, **kwargs)
            elif t not in conclusions:
                new_conclusions = self.start_rules_dict[type(t)].fit_case(x_cp, t, expert, **kwargs)
            if new_conclusions:
                new_conclusions = new_conclusions if isinstance(new_conclusions, list) else [new_conclusions]
                for conclusion in new_conclusions:
                    if type(conclusion) in x_cp and conclusion.mutually_exclusive:
                        x_cp.remove_attribute_equivalent_to_category(t)
                    x_cp.add_attribute_from_category(conclusion)
        return self.classify(x)
