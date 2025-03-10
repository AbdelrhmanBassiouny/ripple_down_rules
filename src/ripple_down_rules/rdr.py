from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy

from matplotlib import pyplot as plt
from ordered_set import OrderedSet
from sqlalchemy.orm import DeclarativeBase as SQLTable, Session, make_transient
from typing_extensions import List, Optional, Dict, Type, Union, Any, Tuple

from .datastructures import Case, MCRDRMode, CallableExpression, Column, create_row, CaseQuery
from .experts import Expert, Human
from .rules import Rule, SingleClassRule, MultiClassTopRule
from .utils import draw_tree, get_attribute_name_from_value, make_set, get_attribute_by_type, copy_case, \
    get_hint_for_attribute


class RippleDownRules(ABC):
    """
    The abstract base class for the ripple down rules classifiers.
    """
    fig: Optional[plt.Figure] = None
    """
    The figure to draw the tree on.
    """
    expert_accepted_conclusions: Optional[List[Column]] = None
    """
    The conclusions that the expert has accepted, such that they are not asked again.
    """

    def __init__(self, start_rule: Optional[Rule] = None, session: Optional[Session] = None):
        """
        :param start_rule: The starting rule for the classifier.
        :param session: The sqlalchemy orm session.
        """
        self.start_rule = start_rule
        self.session = session
        self.fig: Optional[plt.Figure] = None

    def __call__(self, case: Union[Case, SQLTable]) -> Column:
        return self.classify(case)

    @abstractmethod
    def classify(self, case: Union[Case, SQLTable]) -> Optional[Column]:
        """
        Classify a case.

        :param case: The case to classify.
        :return: The category that the case belongs to.
        """
        pass

    @abstractmethod
    def fit_case(self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs) -> Column:
        """
        Fit the RDR on a case, and ask the expert for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category.

        :param case_query: The query containing the case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        pass

    def fit(self, case_queries: List[CaseQuery],
            expert: Optional[Expert] = None,
            n_iter: int = None,
            animate_tree: bool = False,
            **kwargs_for_fit_case):
        """
        Fit the classifier to a batch of cases and categories.

        :param case_queries: The cases and categories to fit the classifier to.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param n_iter: The number of iterations to fit the classifier for.
        :param animate_tree: Whether to draw the tree while fitting the classifier.
        :param kwargs_for_fit_case: The keyword arguments to pass to the fit_case method.
        """
        cases = [case_query.case for case_query in case_queries]
        targets = [case.target for case in case_queries]
        if animate_tree:
            plt.ion()
        i = 0
        stop_iterating = False
        num_rules: int = 0
        while not stop_iterating:
            all_pred = 0
            all_recall = []
            all_precision = []
            if not targets:
                targets = [None] * len(cases)
            for case_query in case_queries:
                case = case_query.case
                target = case_query.target
                if not target:
                    conclusions = self.classify(case) if self.start_rule and self.start_rule.conditions else []
                    target = expert.ask_for_conclusion(case_query, conclusions)
                pred_cat = self.fit_case(case_query, expert=expert, **kwargs_for_fit_case)
                pred_cat = pred_cat if isinstance(pred_cat, list) else [pred_cat]
                target = target if isinstance(target, list) else [target]
                recall = [not yi or (yi in pred_cat) for yi in target]
                y_type = [type(yi) for yi in target]
                precision = [(pred in target) or (type(pred) not in y_type) for pred in pred_cat]
                match = all(recall) and all(precision)
                all_recall.extend(recall)
                all_precision.extend(precision)
                if not match:
                    print(f"Predicted: {pred_cat} but expected: {target}")
                all_pred += int(match)
                if animate_tree and self.start_rule.size > num_rules:
                    num_rules = self.start_rule.size
                    self.update_figures()
                i += 1
                all_predicted = targets and all_pred == len(targets)
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

    @staticmethod
    def case_has_conclusion(case: Union[Case, SQLTable], conclusion_type: Type) -> bool:
        """
        Check if the case has a conclusion.

        :param case: The case to check.
        :param conclusion_type: The target category type to compare the case with.
        :return: Whether the case has a conclusion or not.
        """
        if isinstance(case, SQLTable):
            prop_name, prop_value = get_attribute_by_type(case, conclusion_type)
            if hasattr(prop_value, "__iter__") and not isinstance(prop_value, str):
                return len(prop_value) > 0
            else:
                return prop_value is not None
        else:
            return conclusion_type in case

    @staticmethod
    def copy_case(case: Union[Case, SQLTable]) -> Union[Case, SQLTable]:
        """
        Copy a case.

        :param case: The case to copy.
        :return: The copied case.
        """
        if isinstance(case, SQLTable):
            make_transient(case)
            return case
        else:
            return copy(case)

    @staticmethod
    def convert_to_case_and_get_new_attribute(case: Union[Case, SQLTable],
                                              attribute: Optional[Any] = None) -> Tuple[Union[Case, SQLTable], Any]:
        """
        Convert the case to a row object and get the new attribute.

        :param case: The case to convert.
        :param attribute: The attribute of the case to find a value for.
        :return: The converted case and the new attribute.
        """
        if not isinstance(case, (Case, SQLTable)):
            attribute_name = get_attribute_name_from_value(case, attribute) if attribute else None
            case = create_row(case)
            attribute = getattr(case, attribute_name) if attribute_name else None
        return case, attribute


RDR = RippleDownRules


class SingleClassRDR(RippleDownRules):
    table: Type[SQLTable]
    target_column: Column

    def fit_case(self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs) -> Column:
        """
        Classify a case, and ask the user for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category if provided.

        :param case_query: The case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        expert = expert if expert else Human(session=self.session)
        case, attribute = case_query.case, case_query.attribute
        if case_query.target is None:
            target = expert.ask_for_conclusion(case_query)
        else:
            target = case_query.target

        if not self.start_rule:
            conditions = expert.ask_for_conditions(case, [target])
            self.start_rule = SingleClassRule(conditions, target, corner_case=case)

        pred = self.evaluate(case)

        if pred.conclusion != target:
            conditions = expert.ask_for_conditions(case, [target], pred)
            pred.fit_rule(case, target, conditions=conditions)

        return self.classify(case)

    def classify(self, case: Case) -> Optional[Column]:
        """
        Classify a case by recursively evaluating the rules until a rule fires or the last rule is reached.
        """
        pred = self.evaluate(case)
        return pred.conclusion if pred.fired else None

    def evaluate(self, case: Case) -> SingleClassRule:
        """
        Evaluate the starting rule on a case.
        """
        matched_rule = self.start_rule(case)
        return matched_rule if matched_rule else self.start_rule

    def write_tree_of_rules_as_source_code_to_a_file(self, filename: str):
        """
        Write the tree of rules as source code to a file.
        """
        conclusion = self.start_rule.conclusion
        if isinstance(conclusion, CallableExpression):
            conclusion_types = [conclusion.conclusion_type]
        elif isinstance(conclusion, Column):
            conclusion_types = list(conclusion._value_range)
        else:
            conclusion_types = [type(conclusion)]
        imports = ("from typing_extensions import Union\n"
                   "from ripple_down_rules.datastructures import Case, SQLTable\n")
        if len(conclusion_types) > 1:
            conclusion_name = "Union[" + ", ".join([c.__name__ for c in conclusion_types]) + "]"
        else:
            conclusion_name = conclusion_types[0].__name__
        for conclusion_type in conclusion_types:
            if conclusion_type.__module__ != "builtins":
                imports += f"from {conclusion_type.__module__} import {conclusion_name}\n\n\n"
        func_def = f"def classify_{conclusion_name.lower()}(case: Union[Case, SQLTable]) -> {conclusion_name}:\n"
        with open(filename, "w") as f:
            f.write(imports)
            f.write(func_def)
            self.write_rules_as_source_code_to_file(self.start_rule, f, " " * 4)

    def write_rules_as_source_code_to_file(self, rule: SingleClassRule, file, parent_indent: str = ""):
        """
        Write the rules as source code to a file.
        """
        if rule.conditions:
            file.write(rule.condition_as_source_code(parent_indent))
            if rule.refinement:
                self.write_rules_as_source_code_to_file(rule.refinement, file, parent_indent + "    ")

            file.write(rule.conclusion_as_source_code(parent_indent))

            if rule.alternative:
                self.write_rules_as_source_code_to_file(rule.alternative, file, parent_indent)


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
    conclusions: Optional[List[Column]] = None
    """
    The conclusions that the case belongs to.
    """
    stop_rule_conditions: Optional[CallableExpression] = None
    """
    The conditions of the stopping rule if needed.
    """

    def __init__(self, start_rules: Optional[List[Rule]] = None,
                 mode: MCRDRMode = MCRDRMode.StopOnly, session: Optional[Session] = None):
        """
        :param start_rules: The starting rules for the classifier, these are the rules that are at the top of the tree
        and are always checked, in contrast to the refinement and alternative rules which are only checked if the
        starting rules fire or not.
        :param mode: The mode of the classifier, either StopOnly or StopPlusRule, or StopPlusRuleCombined.
        :param session: The sqlalchemy orm session.
        """
        self.start_rules = [MultiClassTopRule()] if not start_rules else start_rules
        super(MultiClassRDR, self).__init__(self.start_rules[0], session=session)
        self.mode: MCRDRMode = mode

    def classify(self, case: Union[Case, SQLTable]) -> List[Any]:
        evaluated_rule = self.start_rule
        self.conclusions = []
        while evaluated_rule:
            next_rule = evaluated_rule(case)
            if evaluated_rule.fired:
                self.add_conclusion(evaluated_rule)
            evaluated_rule = next_rule
        return self.conclusions

    def fit_case(self, case_query: CaseQuery, expert: Optional[Expert] = None,
                 add_extra_conclusions: bool = False) -> List[Any]:
        """
        Classify a case, and ask the user for stopping rules or classifying rules if the classification is incorrect
         or missing by comparing the case with the target category if provided.

        :param case_query: The query containing the case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions or for extra conclusions.
        :param add_extra_conclusions: Whether to add extra conclusions after classification is done.
        :return: The conclusions that the case belongs to.
        """
        expert = expert if expert else Human(session=self.session)
        case = case_query.case
        if case_query.target is None:
            targets = [expert.ask_for_conclusion(case_query)]
        else:
            targets = [case_query.target]
        self.expert_accepted_conclusions = []
        user_conclusions = []
        for target in targets:
            self.update_start_rule(case, target, expert)
            self.conclusions = []
            self.stop_rule_conditions = None
            evaluated_rule = self.start_rule
            while evaluated_rule:
                next_rule = evaluated_rule(case)
                good_conclusions = targets + user_conclusions + self.expert_accepted_conclusions

                if evaluated_rule.fired:
                    if target and evaluated_rule.conclusion not in good_conclusions:
                        # if self.case_has_conclusion(case, evaluated_rule.conclusion):
                        # Rule fired and conclusion is different from target
                        self.stop_wrong_conclusion_else_add_it(case, target, expert, evaluated_rule,
                                                               add_extra_conclusions)
                    else:
                        # Rule fired and target is correct or there is no target to compare
                        self.add_conclusion(evaluated_rule)

                if not next_rule:
                    if target not in self.conclusions:
                        # Nothing fired and there is a target that should have been in the conclusions
                        self.add_rule_for_case(case, target, expert)
                        # Have to check all rules again to make sure only this new rule fires
                        next_rule = self.start_rule
                    elif add_extra_conclusions and not user_conclusions:
                        # No more conclusions can be made, ask the expert for extra conclusions if needed.
                        user_conclusions.extend(self.ask_expert_for_extra_conclusions(expert, case))
                        if user_conclusions:
                            next_rule = self.last_top_rule
                evaluated_rule = next_rule
        return self.conclusions

    def update_start_rule(self, case: Union[Case, SQLTable], target: Any, expert: Expert):
        """
        Update the starting rule of the classifier.

        :param case: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if not self.start_rule.conditions:
            conditions = expert.ask_for_conditions(case, target)
            self.start_rule.conditions = conditions
            self.start_rule.conclusion = target
            self.start_rule.corner_case = case

    @property
    def last_top_rule(self) -> Optional[MultiClassTopRule]:
        """
        Get the last top rule in the tree.
        """
        if not self.start_rule.furthest_alternative:
            return self.start_rule
        else:
            return self.start_rule.furthest_alternative[-1]

    def stop_wrong_conclusion_else_add_it(self, case: Union[Case, SQLTable], target: Any, expert: Expert,
                                          evaluated_rule: MultiClassTopRule,
                                          add_extra_conclusions: bool):
        """
        Stop a wrong conclusion by adding a stopping rule.
        """
        if self.is_same_category_type(evaluated_rule.conclusion, target) \
                and self.is_conflicting_with_target(evaluated_rule.conclusion, target):
            self.stop_conclusion(case, target, expert, evaluated_rule)
        elif not self.conclusion_is_correct(case, target, expert, evaluated_rule, add_extra_conclusions):
            self.stop_conclusion(case, target, expert, evaluated_rule)

    def stop_conclusion(self, case: Union[Case, SQLTable], target: Any,
                        expert: Expert, evaluated_rule: MultiClassTopRule):
        """
        Stop a conclusion by adding a stopping rule.

        :param case: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param evaluated_rule: The evaluated rule to ask the expert about.
        """
        conditions = expert.ask_for_conditions(case, target, evaluated_rule)
        evaluated_rule.fit_rule(case, target, conditions=conditions)
        if self.mode == MCRDRMode.StopPlusRule:
            self.stop_rule_conditions = conditions
        if self.mode == MCRDRMode.StopPlusRuleCombined:
            new_top_rule_conditions = conditions.combine_with(evaluated_rule.conditions)
            self.add_top_rule(new_top_rule_conditions, target, case)

    @staticmethod
    def is_conflicting_with_target(conclusion: Any, target: Any) -> bool:
        """
        Check if the conclusion is conflicting with the target category.

        :param conclusion: The conclusion to check.
        :param target: The target category to compare the conclusion with.
        :return: Whether the conclusion is conflicting with the target category.
        """
        if hasattr(conclusion, "mutually_exclusive") and conclusion.mutually_exclusive:
            return True
        else:
            return not make_set(conclusion).issubset(make_set(target))

    @staticmethod
    def is_same_category_type(conclusion: Any, target: Any) -> bool:
        """
        Check if the conclusion is of the same class as the target category.

        :param conclusion: The conclusion to check.
        :param target: The target category to compare the conclusion with.
        :return: Whether the conclusion is of the same class as the target category but has a different value.
        """
        return conclusion.__class__ == target.__class__ and target.__class__ != Column

    def conclusion_is_correct(self, case: Union[Case, SQLTable], target: Any, expert: Expert, evaluated_rule: Rule,
                              add_extra_conclusions: bool) -> bool:
        """
        Ask the expert if the conclusion is correct, and add it to the conclusions if it is.

        :param case: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param evaluated_rule: The evaluated rule to ask the expert about.
        :param add_extra_conclusions: Whether adding extra conclusions after classification is allowed.
        :return: Whether the conclusion is correct or not.
        """
        conclusions = list(OrderedSet(self.conclusions))
        if (add_extra_conclusions and expert.ask_if_conclusion_is_correct(case, evaluated_rule.conclusion,
                                                                          targets=target,
                                                                          current_conclusions=conclusions)):
            self.add_conclusion(evaluated_rule)
            self.expert_accepted_conclusions.append(evaluated_rule.conclusion)
            return True
        return False

    def add_rule_for_case(self, case: Union[Case, SQLTable], target: Any, expert: Expert):
        """
        Add a rule for a case that has not been classified with any conclusion.
        """
        if self.stop_rule_conditions and self.mode == MCRDRMode.StopPlusRule:
            conditions = self.stop_rule_conditions
            self.stop_rule_conditions = None
        else:
            conditions = expert.ask_for_conditions(case, target)
        self.add_top_rule(conditions, target, case)

    def ask_expert_for_extra_conclusions(self, expert: Expert, case: Union[Case, SQLTable]) -> List[Any]:
        """
        Ask the expert for extra conclusions when no more conclusions can be made.

        :param expert: The expert to ask for extra conclusions.
        :param case: The case to ask extra conclusions for.
        :return: The extra conclusions that the expert has provided.
        """
        extra_conclusions = []
        conclusions = list(OrderedSet(self.conclusions))
        if not expert.use_loaded_answers:
            print("current conclusions:", conclusions)
        extra_conclusions_dict = expert.ask_for_extra_conclusions(case, conclusions)
        if extra_conclusions_dict:
            for conclusion, conditions in extra_conclusions_dict.items():
                self.add_top_rule(conditions, conclusion, case)
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
            combined_conclusion = evaluated_rule.conclusion if isinstance(evaluated_rule.conclusion, set) \
                else {evaluated_rule.conclusion}
            for c in same_type_conclusions:
                combined_conclusion.update(c if isinstance(c, set) else make_set(c))
                self.conclusions.remove(c)
            self.conclusions.extend(combined_conclusion)

    def add_top_rule(self, conditions: CallableExpression, conclusion: Any, corner_case: Union[Case, SQLTable]):
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

    def __init__(self, category_rdr_map: Optional[Dict[Type, Union[SingleClassRDR, MultiClassRDR]]] = None):
        """
        :param category_rdr_map: A map of categories to ripple down rules classifiers,
        where each category is a parent category that has a set of mutually exclusive (in case of SCRDR) child
        categories, e.g. {Species: SCRDR, Habitat: MCRDR}, where Species and Habitat are parent categories and SCRDR
        and MCRDR are SingleClass and MultiClass ripple down rules classifiers. Species can have child categories like
        Mammal, Bird, Fish, etc. which are mutually exclusive, and Habitat can have child categories like
        Land, Water, Air, etc, which are not mutually exclusive due to some animals living more than one habitat.
        """
        self.start_rules_dict: Dict[Type, Union[SingleClassRDR, MultiClassRDR]] \
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

    def classify(self, case: Union[Case, SQLTable]) -> Optional[List[Any]]:
        """
        Classify a case by going through all RDRs and adding the categories that are classified, and then restarting
        the classification until no more categories can be added.

        :param case: The case to classify.
        :return: The categories that the case belongs to.
        """
        conclusions = []
        case_cp = copy_case(case)
        while True:
            added_attributes = False
            for cat_type, rdr in self.start_rules_dict.items():
                if self.case_has_conclusion(case_cp, cat_type):
                    continue
                pred_atts = rdr.classify(case_cp)
                if pred_atts:
                    pred_atts = pred_atts if isinstance(pred_atts, list) else [pred_atts]
                    pred_atts = [p for p in pred_atts if p not in conclusions]
                    added_attributes = True
                    conclusions.extend(pred_atts)
                    self.update_case_with_same_type_conclusions(case_cp, pred_atts)
            if not added_attributes:
                break
        return conclusions

    def fit_case(self, case_queries: List[CaseQuery], expert: Optional[Expert] = None, **kwargs) -> List[Any]:
        """
        Fit the GRDR on a case, if the target is a new type of category, a new RDR is created for it,
        else the existing RDR of that type will be fitted on the case, and then classification is done and all
        concluded categories are returned. If the category is mutually exclusive, an SCRDR is created, else an MCRDR.
        In case of SCRDR, multiple conclusions of the same type replace each other, in case of MCRDR, they are added if
        they are accepted by the expert, and the attribute of that category is represented in the case as a set of
        values.

        :param case_queries: The queries containing the case to classify and the target categories to compare the case
        with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The categories that the case belongs to.
        """
        expert = expert if expert else Human()
        case_queries = [case_queries] if not isinstance(case_queries, list) else case_queries
        assert len(case_queries) > 0, "No case queries provided"
        case = case_queries[0].case
        assert all([case is case_query.case for case_query in case_queries]), ("fit_case requires only one case,"
                                                                               " for multiple cases use fit instead")
        case_query_cp = copy(case_queries[0])
        case_cp = case_query_cp.case
        for case_query in case_queries:
            target = case_query.target
            if not target:
                target = expert.ask_for_conclusion(case_query)
            case_query_cp = CaseQuery(case_cp, attribute_name=case_query.attribute_name, target=target)
            if type(target) not in self.start_rules_dict:
                conclusions = self.classify(case)
                self.update_case_with_same_type_conclusions(case_cp, conclusions)
                new_rdr = self.initialize_new_rdr_for_attribute(target, case_cp)
                new_conclusions = new_rdr.fit_case(case_query_cp, expert, **kwargs)
                self.start_rules_dict[type(target)] = new_rdr
                self.update_case_with_same_type_conclusions(case_cp, new_conclusions, type(target))
            elif not self.case_has_conclusion(case_cp, type(target)):
                for rdr_type, rdr in self.start_rules_dict.items():
                    if type(target) is not rdr_type:
                        conclusions = rdr.classify(case_cp)
                    else:
                        conclusions = self.start_rules_dict[type(target)].fit_case(case_query_cp,
                                                                                   expert, **kwargs)
                    self.update_case_with_same_type_conclusions(case_cp, conclusions, rdr_type)

        return self.classify(case)

    @staticmethod
    def initialize_new_rdr_for_attribute(attribute: Any, case: Union[Case, SQLTable]):
        """
        Initialize the appropriate RDR type for the target.
        """
        if isinstance(case, SQLTable):
            prop = get_attribute_by_type(case, type(attribute))
            if hasattr(prop, "__iter__") and not isinstance(prop, str):
                return MultiClassRDR()
            else:
                return SingleClassRDR()
        else:
            return SingleClassRDR() if attribute.mutually_exclusive else MultiClassRDR()

    @staticmethod
    def update_case_with_same_type_conclusions(case: Union[Case, SQLTable],
                                               conclusions: List[Any], attribute_type: Optional[Any] = None):
        """
        Update the case with the conclusions.

        :param case: The case to update.
        :param conclusions: The conclusions to update the case with.
        :param attribute_type: The type of the attribute to update.
        """
        if not conclusions:
            return
        conclusions = [conclusions] if not isinstance(conclusions, list) else conclusions
        if len(conclusions) == 0:
            return
        if isinstance(case, SQLTable):
            conclusions_type = type(conclusions[0]) if not attribute_type else attribute_type
            attr_name, attribute = get_attribute_by_type(case, conclusions_type)
            hint, origin, args = get_hint_for_attribute(attr_name, case)
            if isinstance(attribute, set) or origin == set:
                attribute = set() if attribute is None else attribute
                attribute.update(*[make_set(c) for c in conclusions])
            elif isinstance(attribute, list) or origin == list:
                attribute = [] if attribute is None else attribute
                attribute.extend(conclusions)
            elif len(conclusions) == 1 and hint == conclusions_type:
                setattr(case, attr_name, conclusions.pop())
            else:
                raise ValueError(f"Cannot add multiple conclusions to attribute {attr_name}")
        else:
            case.update(*[c.as_dict for c in make_set(conclusions)])

    @property
    def names_of_all_types(self) -> List[str]:
        """
        Get the names of all the types of categories that the GRDR can classify.
        """
        return [t.__name__ for t in self.start_rules_dict.keys()]

    @property
    def all_types(self) -> List[Type]:
        """
        Get all the types of categories that the GRDR can classify.
        """
        return list(self.start_rules_dict.keys())
