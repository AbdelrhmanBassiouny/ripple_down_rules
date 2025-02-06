from __future__ import annotations

from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
from orderedset import OrderedSet
from typing_extensions import List, Optional, Dict, Type, Union

from .datastructures import Category, Condition, Case, Stop, MCRDRMode
from .experts import Expert, Human
from .rules import Rule, SingleClassRule, TopRule, MultiClassStopRule
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
            for x, y in zip(x_batch, y_batch):
                pred_cat = self.fit_case(x, y, expert=expert, **kwargs_for_classify)
                pred_cat = pred_cat if isinstance(pred_cat, list) else [pred_cat]
                match = y in pred_cat
                if not match:
                    print(f"Predicted: {pred_cat[0]} but expected: {y}")
                all_pred += int(match)
                if animate_tree:
                    draw_tree(self.start_rule, self.fig)
                i += 1
                if n_iter and i >= n_iter:
                    break
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

    def classify(self, x: Case) -> List[Category]:
        self.evaluated_rules = []
        for rule in self.start_rules:
            evaluated_rule = rule(x)
            self.evaluated_rules.append(evaluated_rule)
        self.conclusions = [evaluated_rule.conclusion for evaluated_rule in self.evaluated_rules]
        return self.conclusions

    def __init__(self, start_rules: Optional[List[Rule]] = None,
                 mode: MCRDRMode = MCRDRMode.StopOnly):
        """
        :param start_rules: The starting rules for the classifier, these are the rules that are at the top of the tree
        and are always checked, in contrast to the refinement and alternative rules which are only checked if the
        starting rules fire or not.
        :param mode: The mode of the classifier, either StopOnly or StopPlusRule.
        """
        self.start_rules = [TopRule()] if not start_rules else start_rules
        super(MultiClassRDR, self).__init__(self.start_rules[0])
        self.mode: MCRDRMode = mode

    def fit_case(self, x: Case, target: Optional[Category] = None,
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
        if not self.start_rule.conditions:
            conditions = expert.ask_for_conditions(x, target)
            self.start_rule.conditions = conditions
            self.start_rule.conclusion = target
            self.start_rule.corner_case = Case(x.id_, x.attributes_list)

        self.evaluated_rules = []
        self.conclusions = []
        self.expert_accepted_conclusions = []
        self.stop_rule_conditions = None
        user_conclusions = []
        evaluated_rule = self.start_rule
        while evaluated_rule:
            next_rule = evaluated_rule(x)

            if evaluated_rule.fired and evaluated_rule.conclusion != Stop():
                if target and evaluated_rule.conclusion not in [target, *user_conclusions]:
                    # Rule fired and conclusion is different from target
                    self.stop_wrong_conclusion_else_add_it(x, target, expert, evaluated_rule, add_extra_conclusions)
                else:
                    # Rule fired and target is correct or there is no target to compare
                    self.add_conclusion(evaluated_rule)

            if target and not next_rule:
                if target not in self.conclusions:
                    # Nothing fired and there is a target that should have been in the conclusions
                    self.add_rule_for_case(x, target, expert)
                    evaluated_rule = self.start_rule  # Have to check all rules again to make sure only this new rule fires
                    continue
                elif add_extra_conclusions and not user_conclusions:
                    # No more conclusions can be made, ask the expert for extra conclusions if needed.
                    user_conclusions.extend(self.ask_expert_for_extra_conclusions(expert, x))
                    if user_conclusions:
                        evaluated_rule = self.start_rule.furthest_next_rule[0]
                        continue
            evaluated_rule = next_rule
        return list(OrderedSet(self.conclusions))

    def is_last_rule(self, rule_idx: int) -> bool:
        """
        Check if the rule index is the last rule in the classifier.

        :param rule_idx: The index of the rule to check.
        :return: Whether the rule index is the last rule in the classifier.
        """
        return rule_idx == len(self.start_rules) - 1

    def stop_wrong_conclusion_else_add_it(self, x: Case, target: Category, expert: Expert,
                                          evaluated_rule: TopRule,
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
                                                                          target=target,
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
        self.evaluated_rules.append(evaluated_rule)
        self.conclusions.append(evaluated_rule.conclusion)

    def add_top_rule(self, conditions: Dict[str, Condition], conclusion: Category, corner_case: Case):
        """
        Add a top rule to the classifier, which is a rule that is always checked and is part of the start_rules list.

        :param conditions: The conditions of the rule.
        :param conclusion: The conclusion of the rule.
        :param corner_case: The corner case of the rule.
        """
        self.start_rule.next_rule = TopRule(conditions, conclusion, corner_case=corner_case)


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

    def __init__(self, category_scrdr_map: Optional[Dict[Type[Category], SingleClassRDR]] = None):
        """
        :param category_scrdr_map: A map of categories to single class ripple down rules classifiers,
        where each category is a parent category that has a set of mutually exclusive child categories,
        e.g. {Species: SCRDR1, Habitat: SCRDR2}, where Species and Habitat are parent categories and SCRDR1 and SCRDR2
        are single class ripple down rules classifiers. Species can have child categories like Mammal, Bird, Fish, etc.
         which are mutually exclusive, and Habitat can have child categories like Land, Water, Air, etc, which are also
            mutually exclusive.
        """
        self.start_rules_dict = category_scrdr_map

    def fit_case(self, x: Case, target: Optional[Category] = None,
                 expert: Optional[Expert] = None,
                 **kwargs) -> Category:
        pass
