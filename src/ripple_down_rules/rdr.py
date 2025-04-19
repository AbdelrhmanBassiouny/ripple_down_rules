from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from copy import copy
from types import ModuleType

from matplotlib import pyplot as plt
from ordered_set import OrderedSet
from sqlalchemy.orm import DeclarativeBase as SQLTable, Session
from typing_extensions import List, Optional, Dict, Type, Union, Any, Self, Tuple, Callable

from .datastructures import Case, MCRDRMode, CallableExpression, CaseAttribute, CaseQuery
from .experts import Expert, Human
from .rules import Rule, SingleClassRule, MultiClassTopRule, MultiClassStopRule
from .utils import draw_tree, make_set, get_attribute_by_type, copy_case, \
    get_hint_for_attribute, SubclassJSONSerializer, is_iterable, make_list, get_full_class_name, get_type_from_string


class RippleDownRules(SubclassJSONSerializer, ABC):
    """
    The abstract base class for the ripple down rules classifiers.
    """
    fig: Optional[plt.Figure] = None
    """
    The figure to draw the tree on.
    """
    expert_accepted_conclusions: Optional[List[CaseAttribute]] = None
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

    def __call__(self, case: Union[Case, SQLTable]) -> CaseAttribute:
        return self.classify(case)

    @abstractmethod
    def classify(self, case: Union[Case, SQLTable]) -> Optional[CaseAttribute]:
        """
        Classify a case.

        :param case: The case to classify.
        :return: The category that the case belongs to.
        """
        pass

    @abstractmethod
    def fit_case(self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs) \
            -> Union[CaseAttribute, CallableExpression]:
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
            if not targets:
                targets = [None] * len(cases)
            for case_query in case_queries:
                target = case_query.target
                pred_cat = self.fit_case(case_query, expert=expert, **kwargs_for_fit_case)
                match = self.is_matching(pred_cat, target)
                if not match:
                    print(f"Predicted: {pred_cat} but expected: {target}")
                all_pred += int(match)
                if animate_tree and self.start_rule.size > num_rules:
                    num_rules = self.start_rule.size
                    self.update_figures()
            i += 1
            all_predictions = [1 if self.is_matching(self.classify(case), target) else 0
                               for case, target in zip(cases, targets)]
            all_pred = sum(all_predictions)
            print(f"Accuracy: {all_pred}/{len(targets)}")
            all_predicted = targets and all_pred == len(targets)
            num_iter_reached = n_iter and i >= n_iter
            stop_iterating = all_predicted or num_iter_reached
            if stop_iterating:
                break
        print(f"Finished training in {i} iterations")
        if animate_tree:
            plt.ioff()
            plt.show()

    @staticmethod
    def calculate_precision_and_recall(pred_cat: List[CaseAttribute], target: List[CaseAttribute]) -> Tuple[
        List[bool], List[bool]]:
        """
        :param pred_cat: The predicted category.
        :param target: The target category.
        :return: The precision and recall of the classifier.
        """
        pred_cat = pred_cat if is_iterable(pred_cat) else [pred_cat]
        target = target if is_iterable(target) else [target]
        recall = [not yi or (yi in pred_cat) for yi in target]
        target_types = [type(yi) for yi in target]
        precision = [(pred in target) or (type(pred) not in target_types) for pred in pred_cat]
        return precision, recall

    def is_matching(self, pred_cat: List[CaseAttribute], target: List[CaseAttribute]) -> bool:
        """
        :param pred_cat: The predicted category.
        :param target: The target category.
        :return: Whether the classifier is matching or not.
        """
        precision, recall = self.calculate_precision_and_recall(pred_cat, target)
        return all(recall) and all(precision)

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


class RDRWithCodeWriter(RippleDownRules, ABC):

    @abstractmethod
    def write_rules_as_source_code_to_file(self, rule: Rule, file, parent_indent: str = ""):
        """
        Write the rules as source code to a file.

        :param rule: The rule to write as source code.
        :param file: The file to write the source code to.
        :param parent_indent: The indentation of the parent rule.
        """
        pass

    def write_to_python_file(self, file_path: str):
        """
        Write the tree of rules as source code to a file.

        :param file_path: The path to the file to write the source code to.
        """
        func_def = f"def classify(case: {self.case_type.__name__}) -> {self.conclusion_type_hint}:\n"
        with open(file_path + f"/{self.generated_python_file_name}.py", "w") as f:
            f.write(self._get_imports() + "\n\n")
            f.write(func_def)
            f.write(f"{' ' * 4}if not isinstance(case, Case):\n"
                    f"{' ' * 4}    case = create_case(case, recursion_idx=3)\n""")
            self.write_rules_as_source_code_to_file(self.start_rule, f, " " * 4)

    @property
    @abstractmethod
    def conclusion_type_hint(self) -> str:
        """
        :return: The type hint of the conclusion of the rdr as a string.
        """
        pass

    def _get_imports(self) -> str:
        """
        :return: The imports for the generated python file of the RDR as a string.
        """
        imports = ""
        if self.case_type.__module__ != "builtins":
            imports += f"from {self.case_type.__module__} import {self.case_type.__name__}\n"
        if self.conclusion_type.__module__ != "builtins":
            imports += f"from {self.conclusion_type.__module__} import {self.conclusion_type.__name__}\n"
        imports += "from ripple_down_rules.datastructures import Case, create_case\n"
        return imports

    def get_rdr_classifier_from_python_file(self, package_name) -> Callable[[Any], Any]:
        """
        :param package_name: The name of the package that contains the RDR classifier function.
        :return: The module that contains the rdr classifier function.
        """
        return importlib.import_module(f"{package_name.strip('./')}.{self.generated_python_file_name}").classify

    @property
    def generated_python_file_name(self) -> str:
        return f"{self.conclusion_type.__name__.lower()}_{self.__class__.__name__}"

    @property
    def python_file_name(self):
        return f"{self.start_rule.conclusion.__name__.lower()}_rdr"

    @property
    def case_type(self) -> Type:
        """
        :return: The type of the case (input) to the RDR classifier.
        """
        if isinstance(self.start_rule.corner_case, Case):
            return self.start_rule.corner_case._type
        else:
            return type(self.start_rule.corner_case)

    @property
    def conclusion_type(self) -> Type:
        """
        :return: The type of the conclusion of the RDR classifier.
        """
        if isinstance(self.start_rule.conclusion, CallableExpression):
            return self.start_rule.conclusion.conclusion_type
        else:
            if isinstance(self.start_rule.conclusion, set):
                return type(list(self.start_rule.conclusion)[0])
            return type(self.start_rule.conclusion)


class SingleClassRDR(RDRWithCodeWriter):

    def fit_case(self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs) \
            -> Union[CaseAttribute, CallableExpression]:
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
        target_name = case_query.attribute_name if case_query.attribute_name else type(target).__name__
        target_dict = {target_name: target}
        if not self.start_rule:
            conditions = expert.ask_for_conditions(case, target_dict)
            self.start_rule = SingleClassRule(conditions, target, corner_case=case)

        pred = self.evaluate(case)

        if pred.conclusion != target:
            conditions = expert.ask_for_conditions(case, target_dict, pred)
            pred.fit_rule(case, target, conditions=conditions)

        return self.classify(case)

    def classify(self, case: Case) -> Optional[CaseAttribute]:
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

    def write_rules_as_source_code_to_file(self, rule: SingleClassRule, file, parent_indent: str = ""):
        """
        Write the rules as source code to a file.
        """
        if rule.conditions:
            file.write(rule.write_condition_as_source_code(parent_indent))
            if rule.refinement:
                self.write_rules_as_source_code_to_file(rule.refinement, file, parent_indent + "    ")

            file.write(rule.write_conclusion_as_source_code(parent_indent))

            if rule.alternative:
                self.write_rules_as_source_code_to_file(rule.alternative, file, parent_indent)

    @property
    def conclusion_type_hint(self) -> str:
        return self.conclusion_type.__name__

    def _to_json(self) -> Dict[str, Any]:
        return {"start_rule": self.start_rule.to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create an instance of the class from a json
        """
        start_rule = SingleClassRule.from_json(data["start_rule"])
        return cls(start_rule)


class MultiClassRDR(RDRWithCodeWriter):
    """
    A multi class ripple down rules classifier, which can draw multiple conclusions for a case.
    This is done by going through all rules and checking if they fire or not, and adding stopping rules if needed,
    when wrong conclusions are made to stop these rules from firing again for similar cases.
    """
    evaluated_rules: Optional[List[Rule]] = None
    """
    The evaluated rules in the classifier for one case.
    """
    conclusions: Optional[List[CaseAttribute]] = None
    """
    The conclusions that the case belongs to.
    """
    stop_rule_conditions: Optional[CallableExpression] = None
    """
    The conditions of the stopping rule if needed.
    """

    def __init__(self, start_rule: Optional[Rule] = None,
                 mode: MCRDRMode = MCRDRMode.StopOnly, session: Optional[Session] = None):
        """
        :param start_rule: The starting rules for the classifier.
        :param mode: The mode of the classifier, either StopOnly or StopPlusRule, or StopPlusRuleCombined.
        :param session: The sqlalchemy orm session.
        """
        start_rule = MultiClassTopRule() if not start_rule else start_rule
        super(MultiClassRDR, self).__init__(start_rule, session=session)
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
                 add_extra_conclusions: bool = False) -> List[Union[CaseAttribute, CallableExpression]]:
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
        target_names = [case_query.attribute_name] if case_query.attribute_name else \
            [type(target).__name__ for target in targets]
        targets = dict(zip(target_names, targets))
        self.expert_accepted_conclusions = []
        user_conclusions = []
        for target_name, target_value in targets.items():
            target = {target_name: target_value}
            self.update_start_rule(case, target, expert)
            self.conclusions = []
            self.stop_rule_conditions = None
            evaluated_rule = self.start_rule
            while evaluated_rule:
                next_rule = evaluated_rule(case)
                good_conclusions = list(targets.values()) + user_conclusions + self.expert_accepted_conclusions

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
                    if not make_set(list(target.values())[0]).intersection(make_set(self.conclusions)):
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

    def write_rules_as_source_code_to_file(self, rule: Union[MultiClassTopRule, MultiClassStopRule],
                                           file, parent_indent: str = ""):
        """
        Write the rules as source code to a file.

        :
        """
        if rule == self.start_rule:
            file.write(f"{parent_indent}conclusions = set()\n")
        if rule.conditions:
            file.write(rule.write_condition_as_source_code(parent_indent))
            conclusion_indent = parent_indent
            if hasattr(rule, "refinement") and rule.refinement:
                self.write_rules_as_source_code_to_file(rule.refinement, file, parent_indent + "    ")
                conclusion_indent = parent_indent + " " * 4
                file.write(f"{conclusion_indent}else:\n")
            file.write(rule.write_conclusion_as_source_code(conclusion_indent))

            if rule.alternative:
                self.write_rules_as_source_code_to_file(rule.alternative, file, parent_indent)

    @property
    def conclusion_type_hint(self) -> str:
        return f"Set[{self.conclusion_type.__name__}]"

    def _get_imports(self) -> str:
        imports = super()._get_imports()
        imports += "from typing_extensions import Set\n"
        return imports

    def update_start_rule(self, case: Union[Case, SQLTable], target: Dict[str, Any], expert: Expert):
        """
        Update the starting rule of the classifier.

        :param case: The case to classify.
        :param target: The target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if not self.start_rule.conditions:
            conditions = expert.ask_for_conditions(case, target)
            self.start_rule.conditions = conditions
            self.start_rule.conclusion = list(target.values())[0]
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

    def stop_wrong_conclusion_else_add_it(self, case: Union[Case, SQLTable], target: Dict[str, Any], expert: Expert,
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

    def stop_conclusion(self, case: Union[Case, SQLTable], target: Dict[str, Any],
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
            self.add_top_rule(new_top_rule_conditions, list(target.values())[0], case)

    @staticmethod
    def is_conflicting_with_target(conclusion: Any, target: Dict[str, Any]) -> bool:
        """
        Check if the conclusion is conflicting with the target category.

        :param conclusion: The conclusion to check.
        :param target: The target category to compare the conclusion with.
        :return: Whether the conclusion is conflicting with the target category.
        """
        if hasattr(conclusion, "mutually_exclusive") and conclusion.mutually_exclusive:
            return True
        else:
            return not make_set(conclusion).issubset(make_set(target.values()))

    @staticmethod
    def is_same_category_type(conclusion: Any, target: Dict[str, Any]) -> bool:
        """
        Check if the conclusion is of the same class as the target category.

        :param conclusion: The conclusion to check.
        :param target: The target category to compare the conclusion with.
        :return: Whether the conclusion is of the same class as the target category but has a different value.
        """
        target = list(target.values())[0]
        return conclusion.__class__ == target.__class__ and target.__class__ != CaseAttribute

    def conclusion_is_correct(self, case: Union[Case, SQLTable], target: Dict[str, Any],
                              expert: Expert, evaluated_rule: Rule,
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
        conclusions = {list(target.keys())[0]: c for c in OrderedSet(self.conclusions)}
        if (add_extra_conclusions and expert.ask_if_conclusion_is_correct(case, evaluated_rule.conclusion,
                                                                          targets=target,
                                                                          current_conclusions=conclusions)):
            self.add_conclusion(evaluated_rule)
            self.expert_accepted_conclusions.append(evaluated_rule.conclusion)
            return True
        return False

    def add_rule_for_case(self, case: Union[Case, SQLTable], target: Dict[str, Any], expert: Expert):
        """
        Add a rule for a case that has not been classified with any conclusion.
        """
        if self.stop_rule_conditions and self.mode == MCRDRMode.StopPlusRule:
            conditions = self.stop_rule_conditions
            self.stop_rule_conditions = None
        else:
            conditions = expert.ask_for_conditions(case, target)
        self.add_top_rule(conditions, list(target.values())[0], case)

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
            self.conclusions.extend(make_list(evaluated_rule.conclusion))
        else:
            same_type_conclusions = [c for c in self.conclusions if type(c) == type(evaluated_rule.conclusion)]
            combined_conclusion = evaluated_rule.conclusion if isinstance(evaluated_rule.conclusion, set) \
                else {evaluated_rule.conclusion}
            combined_conclusion = copy(combined_conclusion)
            for c in same_type_conclusions:
                combined_conclusion.update(c if isinstance(c, set) else make_set(c))
                self.conclusions.remove(c)
            self.conclusions.extend(make_list(combined_conclusion))

    def add_top_rule(self, conditions: CallableExpression, conclusion: Any, corner_case: Union[Case, SQLTable]):
        """
        Add a top rule to the classifier, which is a rule that is always checked and is part of the start_rules list.

        :param conditions: The conditions of the rule.
        :param conclusion: The conclusion of the rule.
        :param corner_case: The corner case of the rule.
        """
        self.start_rule.alternative = MultiClassTopRule(conditions, conclusion, corner_case=corner_case)

    def _to_json(self) -> Dict[str, Any]:
        return {"start_rule": self.start_rule.to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create an instance of the class from a json
        """
        start_rule = SingleClassRule.from_json(data["start_rule"])
        return cls(start_rule)


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
            self.start_rules_dict[value.conclusion_type] = value

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
        return self._classify(self.start_rules_dict, case)

    @staticmethod
    def _classify(classifiers_dict: Dict[Type, Union[ModuleType, RippleDownRules]],
                  case: Union[Case, SQLTable]) -> Optional[List[Any]]:
        """
        Classify a case by going through all classifiers and adding the categories that are classified,
         and then restarting the classification until no more categories can be added.

        :param classifiers_dict: A dictionary mapping conclusion types to the classifiers that produce them.
        :param case: The case to classify.
        :return: The categories that the case belongs to.
        """
        conclusions = []
        case_cp = copy_case(case)
        while True:
            added_attributes = False
            for cat_type, rdr in classifiers_dict.items():
                if GeneralRDR.case_has_conclusion(case_cp, cat_type):
                    continue
                pred_atts = rdr.classify(case_cp)
                if pred_atts:
                    pred_atts = make_list(pred_atts)
                    pred_atts = [p for p in pred_atts if p not in conclusions]
                    added_attributes = True
                    conclusions.extend(pred_atts)
                    GeneralRDR.update_case(case_cp, pred_atts)
            if not added_attributes:
                break
        return conclusions

    def fit_case(self, case_queries: List[CaseQuery], expert: Optional[Expert] = None, **kwargs) \
            -> List[Union[CaseAttribute, CallableExpression]]:
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
                conclusions = self.classify(case) if self.start_rule and self.start_rule.conditions else []
                target = expert.ask_for_conclusion(case_query, conclusions)
            case_query_cp = CaseQuery(case_cp, attribute_name=case_query.attribute_name, target=target)
            if is_iterable(target) and not isinstance(target, CaseAttribute):
                target_type = type(make_list(target)[0])
                assert all([type(t) is target_type for t in target]), ("All targets of a case query must be of the same"
                                                                       " type")
            else:
                target_type = type(target)
            if target_type not in self.start_rules_dict:
                conclusions = self.classify(case)
                self.update_case(case_cp, conclusions)
                new_rdr = self.initialize_new_rdr_for_attribute(target, case_cp)
                new_conclusions = new_rdr.fit_case(case_query_cp, expert, **kwargs)
                self.start_rules_dict[target_type] = new_rdr
                self.update_case(case_cp, new_conclusions, target_type)
            elif not self.case_has_conclusion(case_cp, target_type):
                for rdr_type, rdr in self.start_rules_dict.items():
                    if target_type is not rdr_type:
                        conclusions = rdr.classify(case_cp)
                    else:
                        conclusions = self.start_rules_dict[target_type].fit_case(case_query_cp,
                                                                                  expert, **kwargs)
                    self.update_case(case_cp, conclusions, rdr_type)

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
        elif isinstance(attribute, CaseAttribute):
            return SingleClassRDR() if attribute.mutually_exclusive else MultiClassRDR()
        else:
            return MultiClassRDR() if is_iterable(attribute) else SingleClassRDR()

    @staticmethod
    def update_case(case: Union[Case, SQLTable],
                    conclusions: List[Any], attribute_type: Optional[Any] = None):
        """
        Update the case with the conclusions.

        :param case: The case to update.
        :param conclusions: The conclusions to update the case with.
        :param attribute_type: The type of the attribute to update.
        """
        if not conclusions:
            return
        conclusions = [conclusions] if not isinstance(conclusions, list) else list(conclusions)
        if len(conclusions) == 0:
            return
        if isinstance(case, SQLTable):
            conclusions_type = type(conclusions[0]) if not attribute_type else attribute_type
            attr_name, attribute = get_attribute_by_type(case, conclusions_type)
            hint, origin, args = get_hint_for_attribute(attr_name, case)
            if isinstance(attribute, set) or origin == set:
                attribute = set() if attribute is None else attribute
                for c in conclusions:
                    attribute.update(make_set(c))
            elif isinstance(attribute, list) or origin == list:
                attribute = [] if attribute is None else attribute
                attribute.extend(conclusions)
            elif len(conclusions) == 1 and hint == conclusions_type:
                setattr(case, attr_name, conclusions.pop())
            else:
                raise ValueError(f"Cannot add multiple conclusions to attribute {attr_name}")
        else:
            for c in make_set(conclusions):
                case.update(c.as_dict)

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

    def _to_json(self) -> Dict[str, Any]:
        return {"start_rules": {get_full_class_name(t): rdr.to_json() for t, rdr in self.start_rules_dict.items()}}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> GeneralRDR:
        """
        Create an instance of the class from a json
        """
        start_rules_dict = {}
        for k, v in data["start_rules"].items():
            k = get_type_from_string(k)
            start_rules_dict[k] = get_type_from_string(v['_type']).from_json(v)
        return cls(start_rules_dict)

    def write_to_python_file(self, file_path: str):
        """
        Write the tree of rules as source code to a file.

        :param file_path: The path to the file to write the source code to.
        """
        for rdr in self.start_rules_dict.values():
            rdr.write_to_python_file(file_path)
        func_def = f"def classify(case: {self.case_type.__name__}) -> {self.conclusion_type_hint}:\n"
        with open(file_path + f"/{self.generated_python_file_name}.py", "w") as f:
            f.write(self._get_imports(file_path) + "\n\n")
            f.write("classifiers_dict = dict()\n")
            for t, rdr in self.start_rules_dict.items():
                f.write(f"classifiers_dict[{t.__name__}] = {t.__name__.lower()}_classifier\n")
            f.write("\n\n")
            f.write(func_def)
            f.write(f"{' ' * 4}if not isinstance(case, Case):\n"
                    f"{' ' * 4}    case = create_case(case, recursion_idx=3)\n""")
            f.write(f"{' ' * 4}return GeneralRDR._classify(classifiers_dict, case)\n")

    @property
    def case_type(self) -> Type:
        """
        :return: The type of the case (input) to the RDR classifier.
        """
        if isinstance(self.start_rule.corner_case, Case):
            return self.start_rule.corner_case._type
        else:
            return type(self.start_rule.corner_case)

    def get_rdr_classifier_from_python_file(self, file_path: str):
        """
        :param file_path: The path to the file that contains the RDR classifier function.
        :return: The module that contains the rdr classifier function.
        """
        return importlib.import_module(f"{file_path.strip('./')}.{self.generated_python_file_name}").classify

    @property
    def generated_python_file_name(self) -> str:
        return f"{self.case_type.__name__.lower()}_grdr"

    @property
    def conclusion_type_hint(self) -> str:
        return f"List[Union[{', '.join([rdr.conclusion_type_hint for rdr in self.start_rules_dict.values()])}]]"

    def _get_imports(self, file_path: str) -> str:
        imports = ""
        # add type hints
        imports += f"from typing_extensions import List, Union, Set\n"
        # import rdr type
        imports += f"from ripple_down_rules.rdr import GeneralRDR\n"
        # add case type
        imports += f"from ripple_down_rules.datastructures import Case, create_case\n"
        imports += f"from {self.case_type.__module__} import {self.case_type.__name__}\n"
        # add conclusion type imports
        for conclusion_type in self.start_rules_dict.keys():
            imports += f"from {conclusion_type.__module__} import {conclusion_type.__name__}\n"
        # add rdr python generated functions.
        for conclusion_type, rdr in self.start_rules_dict.items():
            imports += (f"from {file_path.strip('./')}"
                        f" import {rdr.generated_python_file_name} as {conclusion_type.__name__.lower()}_classifier\n")
        return imports
