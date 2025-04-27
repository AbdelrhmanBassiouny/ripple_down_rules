from __future__ import annotations

import importlib
import re
import sys
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import is_dataclass
from io import TextIOWrapper
from types import ModuleType

from matplotlib import pyplot as plt
from ordered_set import OrderedSet
from sqlalchemy.orm import DeclarativeBase as SQLTable
from typing_extensions import List, Optional, Dict, Type, Union, Any, Self, Tuple, Callable, Set

from .datastructures.case import Case, CaseAttribute
from .datastructures.callable_expression import CallableExpression
from .datastructures.dataclasses import CaseQuery
from .datastructures.enums import MCRDRMode
from .experts import Expert, Human
from .rules import Rule, SingleClassRule, MultiClassTopRule, MultiClassStopRule
from .utils import draw_tree, make_set, copy_case, \
    get_hint_for_attribute, SubclassJSONSerializer, is_iterable, make_list, get_type_from_string, \
    get_case_attribute_type, ask_llm, is_matching


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
    _generated_python_file_name: Optional[str] = None
    """
    The name of the generated python file.
    """

    def __init__(self, start_rule: Optional[Rule] = None):
        """
        :param start_rule: The starting rule for the classifier.
        """
        self.start_rule = start_rule
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
        targets = []
        if animate_tree:
            plt.ion()
        i = 0
        stop_iterating = False
        num_rules: int = 0
        while not stop_iterating:
            for case_query in case_queries:
                pred_cat = self.fit_case(case_query, expert=expert, **kwargs_for_fit_case)
                if case_query.target is None:
                    continue
                target = {case_query.attribute_name: case_query.target(case_query.case)}
                if len(targets) < len(case_queries):
                    targets.append(target)
                match = is_matching(self.classify, case_query, pred_cat)
                if not match:
                    print(f"Predicted: {pred_cat} but expected: {target}")
                if animate_tree and self.start_rule.size > num_rules:
                    num_rules = self.start_rule.size
                    self.update_figures()
            i += 1
            all_predictions = [1 if is_matching(self.classify, case_query) else 0 for case_query in case_queries
                               if case_query.target is not None]
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

    def update_figures(self):
        """
        Update the figures of the classifier.
        """
        if isinstance(self, GeneralRDR):
            for i, (rdr_name, rdr) in enumerate(self.start_rules_dict.items()):
                if not rdr.fig:
                    rdr.fig = plt.figure(f"Rule {i}: {rdr_name}")
                draw_tree(rdr.start_rule, rdr.fig)
        else:
            if not self.fig:
                self.fig = plt.figure(0)
            draw_tree(self.start_rule, self.fig)

    @staticmethod
    def case_has_conclusion(case: Union[Case, SQLTable], conclusion_name: str) -> bool:
        """
        Check if the case has a conclusion.

        :param case: The case to check.
        :param conclusion_name: The target category name to compare the case with.
        :return: Whether the case has a conclusion or not.
        """
        return hasattr(case, conclusion_name) and getattr(case, conclusion_name) is not None

    @property
    def type_(self):
        return self.__class__


class RDRWithCodeWriter(RippleDownRules, ABC):

    @abstractmethod
    def write_rules_as_source_code_to_file(self, rule: Rule, file, parent_indent: str = "",
                                           defs_file: Optional[str] = None):
        """
        Write the rules as source code to a file.

        :param rule: The rule to write as source code.
        :param file: The file to write the source code to.
        :param parent_indent: The indentation of the parent rule.
        :param defs_file: The file to write the definitions to.
        """
        pass

    def write_to_python_file(self, file_path: str, postfix: str = ""):
        """
        Write the tree of rules as source code to a file.

        :param file_path: The path to the file to write the source code to.
        :param postfix: The postfix to add to the file name.
        """
        self.generated_python_file_name = self._default_generated_python_file_name + postfix
        func_def = f"def classify(case: {self.case_type.__name__}) -> {self.conclusion_type_hint}:\n"
        file_name = file_path + f"/{self.generated_python_file_name}.py"
        defs_file_name = file_path + f"/{self.generated_python_defs_file_name}.py"
        imports = self._get_imports()
        # clear the files first
        with open(defs_file_name, "w") as f:
            f.write(imports + "\n\n")
        with open(file_name, "w") as f:
            imports += f"from .{self.generated_python_defs_file_name} import *\n"
            imports += f"from ripple_down_rules.rdr import {self.__class__.__name__}\n"
            f.write(imports + "\n\n")
            f.write(f"conclusion_type = ({', '.join([ct.__name__ for ct in self.conclusion_type])},)\n\n")
            f.write(f"type_ = {self.__class__.__name__}\n\n")
            f.write(func_def)
            f.write(f"{' ' * 4}if not isinstance(case, Case):\n"
                    f"{' ' * 4}    case = create_case(case, max_recursion_idx=3)\n""")
            self.write_rules_as_source_code_to_file(self.start_rule, f, " " * 4, defs_file=defs_file_name)

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
        for conclusion_type in self.conclusion_type:
            if conclusion_type.__module__ != "builtins":
                imports += f"from {conclusion_type.__module__} import {conclusion_type.__name__}\n"
        imports += "from ripple_down_rules.datastructures.case import Case, create_case\n"
        for rule in [self.start_rule] + list(self.start_rule.descendants):
            if not rule.conditions:
                continue
            if rule.conditions.scope is None or len(rule.conditions.scope) == 0:
                continue
            for k, v in rule.conditions.scope.items():
                new_imports = f"from {v.__module__} import {v.__name__}\n"
                if new_imports in imports:
                    continue
                imports += new_imports
        return imports

    def get_rdr_classifier_from_python_file(self, package_name: str) -> Callable[[Any], Any]:
        """
        :param package_name: The name of the package that contains the RDR classifier function.
        :return: The module that contains the rdr classifier function.
        """
        # remove from imports if exists first
        name = f"{package_name.strip('./')}.{self.generated_python_file_name}"
        try:
            module = importlib.import_module(name)
            del sys.modules[name]
        except ModuleNotFoundError:
            pass
        return importlib.import_module(name).classify

    @property
    def generated_python_file_name(self) -> str:
        if self._generated_python_file_name is None:
            self._generated_python_file_name = self._default_generated_python_file_name
        return self._generated_python_file_name

    @generated_python_file_name.setter
    def generated_python_file_name(self, value: str):
        """
        Set the generated python file name.
        :param value: The new value for the generated python file name.
        """
        self._generated_python_file_name = value

    @property
    def _default_generated_python_file_name(self) -> str:
        """
        :return: The default generated python file name.
        """
        return f"{self.start_rule.corner_case._name.lower()}_{self.attribute_name}_{self.acronym.lower()}"

    @property
    def generated_python_defs_file_name(self) -> str:
        return f"{self.generated_python_file_name}_defs"

    @property
    def acronym(self) -> str:
        """
        :return: The acronym of the classifier.
        """
        if self.__class__.__name__ == "GeneralRDR":
            return "GRDR"
        elif self.__class__.__name__ == "MultiClassRDR":
            return "MCRDR"
        else:
            return "SCRDR"


    @property
    def case_type(self) -> Type:
        """
        :return: The type of the case (input) to the RDR classifier.
        """
        if isinstance(self.start_rule.corner_case, Case):
            return self.start_rule.corner_case._obj_type
        else:
            return type(self.start_rule.corner_case)

    @property
    def conclusion_type(self) -> Tuple[Type]:
        """
        :return: The type of the conclusion of the RDR classifier.
        """
        if isinstance(self.start_rule.conclusion, CallableExpression):
            return self.start_rule.conclusion.conclusion_type
        else:
            conclusion = self.start_rule.conclusion
            if isinstance(conclusion, set):
                return type(list(conclusion)[0]), set
            return (type(conclusion),)

    @property
    def attribute_name(self) -> str:
        """
        :return: The name of the attribute that the classifier is classifying.
        """
        return self.start_rule.conclusion_name


class SingleClassRDR(RDRWithCodeWriter):

    def __init__(self, start_rule: Optional[SingleClassRule] = None, default_conclusion: Optional[Any] = None):
        """
        :param start_rule: The starting rule for the classifier.
        :param default_conclusion: The default conclusion for the classifier if no rules fire.
        """
        super(SingleClassRDR, self).__init__(start_rule)
        self.default_conclusion: Optional[Any] = default_conclusion

    def fit_case(self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs) \
            -> Union[CaseAttribute, CallableExpression, None]:
        """
        Classify a case, and ask the user for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category if provided.

        :param case_query: The case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        expert = expert if expert else Human()
        if case_query.default_value is not None and self.default_conclusion != case_query.default_value:
            self.default_conclusion = case_query.default_value
        case = case_query.case
        target = expert.ask_for_conclusion(case_query) if case_query.target is None else case_query.target
        if target is None:
            return self.classify(case)
        if not self.start_rule:
            conditions = expert.ask_for_conditions(case_query)
            self.start_rule = SingleClassRule(conditions, target, corner_case=case,
                                              conclusion_name=case_query.attribute_name)

        pred = self.evaluate(case_query.case)
        if pred.conclusion(case) != target(case):
            conditions = expert.ask_for_conditions(case_query, pred)
            pred.fit_rule(case_query.case, target, conditions=conditions)

        return self.classify(case_query.case)

    def classify(self, case: Case) -> Optional[Any]:
        """
        Classify a case by recursively evaluating the rules until a rule fires or the last rule is reached.
        """
        pred = self.evaluate(case)
        return pred.conclusion(case) if pred.fired else self.default_conclusion

    def evaluate(self, case: Case) -> SingleClassRule:
        """
        Evaluate the starting rule on a case.
        """
        matched_rule = self.start_rule(case)
        return matched_rule if matched_rule else self.start_rule

    def write_to_python_file(self, file_path: str, postfix: str = ""):
        super().write_to_python_file(file_path, postfix)
        if self.default_conclusion is not None:
            with open(file_path + f"/{self.generated_python_file_name}.py", "a") as f:
                f.write(f"{' '*4}else:\n{' '*8}return {self.default_conclusion}\n")

    def write_rules_as_source_code_to_file(self, rule: SingleClassRule, file: TextIOWrapper, parent_indent: str = "",
                                           defs_file: Optional[str] = None):
        """
        Write the rules as source code to a file.
        """
        if rule.conditions:
            if_clause = rule.write_condition_as_source_code(parent_indent, defs_file)
            file.write(if_clause)
            if rule.refinement:

                self.write_rules_as_source_code_to_file(rule.refinement, file, parent_indent + "    ",
                                                        defs_file=defs_file)

            file.write(rule.write_conclusion_as_source_code(parent_indent))

            if rule.alternative:
                self.write_rules_as_source_code_to_file(rule.alternative, file, parent_indent, defs_file=defs_file)

    @property
    def conclusion_type_hint(self) -> str:
        return self.conclusion_type[0].__name__

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
                 mode: MCRDRMode = MCRDRMode.StopOnly):
        """
        :param start_rule: The starting rules for the classifier.
        :param mode: The mode of the classifier, either StopOnly or StopPlusRule, or StopPlusRuleCombined.
        """
        start_rule = MultiClassTopRule() if not start_rule else start_rule
        super(MultiClassRDR, self).__init__(start_rule)
        self.mode: MCRDRMode = mode

    def classify(self, case: Union[Case, SQLTable]) -> Set[Any]:
        evaluated_rule = self.start_rule
        self.conclusions = []
        while evaluated_rule:
            next_rule = evaluated_rule(case)
            if evaluated_rule.fired:
                self.add_conclusion(evaluated_rule, case)
            evaluated_rule = next_rule
        return make_set(self.conclusions)

    def fit_case(self, case_query: CaseQuery, expert: Optional[Expert] = None,
                 add_extra_conclusions: bool = False) -> Set[Union[CaseAttribute, CallableExpression, None]]:
        """
        Classify a case, and ask the user for stopping rules or classifying rules if the classification is incorrect
         or missing by comparing the case with the target category if provided.

        :param case_query: The query containing the case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions or for extra conclusions.
        :param add_extra_conclusions: Whether to add extra conclusions after classification is done.
        :return: The conclusions that the case belongs to.
        """
        expert = expert if expert else Human()
        if case_query.target is None:
            expert.ask_for_conclusion(case_query)
        if case_query.target is None:
            return self.classify(case_query.case)
        self.update_start_rule(case_query, expert)
        self.expert_accepted_conclusions = []
        user_conclusions = []
        self.conclusions = []
        self.stop_rule_conditions = None
        evaluated_rule = self.start_rule
        target = case_query.target(case_query.case)
        while evaluated_rule:
            next_rule = evaluated_rule(case_query.case)
            rule_conclusion = evaluated_rule.conclusion(case_query.case)
            good_conclusions = make_list(target) + user_conclusions + self.expert_accepted_conclusions
            good_conclusions = make_set(good_conclusions)

            if evaluated_rule.fired:
                if target and not make_set(rule_conclusion).issubset(good_conclusions):
                    # Rule fired and conclusion is different from target
                    self.stop_wrong_conclusion_else_add_it(case_query, expert, evaluated_rule,
                                                           add_extra_conclusions)
                else:
                    # Rule fired and target is correct or there is no target to compare
                    self.add_conclusion(evaluated_rule, case_query.case)

            if not next_rule:
                if not make_set(target).issubset(make_set(self.conclusions)):
                    # Nothing fired and there is a target that should have been in the conclusions
                    self.add_rule_for_case(case_query, expert)
                    # Have to check all rules again to make sure only this new rule fires
                    next_rule = self.start_rule
                elif add_extra_conclusions and not user_conclusions:
                    # No more conclusions can be made, ask the expert for extra conclusions if needed.
                    user_conclusions.extend(self.ask_expert_for_extra_conclusions(expert, case_query.case))
                    if user_conclusions:
                        next_rule = self.last_top_rule
            evaluated_rule = next_rule
        return self.conclusions

    def write_rules_as_source_code_to_file(self, rule: Union[MultiClassTopRule, MultiClassStopRule],
                                           file, parent_indent: str = "", defs_file: Optional[str] = None):
        if rule == self.start_rule:
            file.write(f"{parent_indent}conclusions = set()\n")
        if rule.conditions:
            if_clause = rule.write_condition_as_source_code(parent_indent, defs_file)
            file.write(if_clause)
            conclusion_indent = parent_indent
            if hasattr(rule, "refinement") and rule.refinement:
                self.write_rules_as_source_code_to_file(rule.refinement, file, parent_indent + "    ",
                                                        defs_file=defs_file)
                conclusion_indent = parent_indent + " " * 4
                file.write(f"{conclusion_indent}else:\n")
            file.write(rule.write_conclusion_as_source_code(conclusion_indent))

            if rule.alternative:
                self.write_rules_as_source_code_to_file(rule.alternative, file, parent_indent, defs_file=defs_file)

    @property
    def conclusion_type_hint(self) -> str:
        return f"Set[{self.conclusion_type[0].__name__}]"

    def _get_imports(self) -> str:
        imports = super()._get_imports()
        imports += "from typing_extensions import Set\n"
        imports += "from ripple_down_rules.utils import make_set\n"
        return imports

    def update_start_rule(self, case_query: CaseQuery, expert: Expert):
        """
        Update the starting rule of the classifier.

        :param case_query: The case query to update the starting rule with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if not self.start_rule.conditions:
            conditions = expert.ask_for_conditions(case_query)
            self.start_rule.conditions = conditions
            self.start_rule.conclusion = case_query.target
            self.start_rule.corner_case = case_query.case
            self.start_rule.conclusion_name = case_query.attribute_name

    @property
    def last_top_rule(self) -> Optional[MultiClassTopRule]:
        """
        Get the last top rule in the tree.
        """
        if not self.start_rule.furthest_alternative:
            return self.start_rule
        else:
            return self.start_rule.furthest_alternative[-1]

    def stop_wrong_conclusion_else_add_it(self, case_query: CaseQuery, expert: Expert,
                                          evaluated_rule: MultiClassTopRule,
                                          add_extra_conclusions: bool):
        """
        Stop a wrong conclusion by adding a stopping rule.
        """
        target = case_query.target(case_query.case)
        rule_conclusion = evaluated_rule.conclusion(case_query.case)
        if self.is_same_category_type(rule_conclusion, target) \
                and self.is_conflicting_with_target(rule_conclusion, target):
            self.stop_conclusion(case_query, expert, evaluated_rule)
        elif not self.conclusion_is_correct(case_query, expert, evaluated_rule, add_extra_conclusions):
            self.stop_conclusion(case_query, expert, evaluated_rule)

    def stop_conclusion(self, case_query: CaseQuery,
                        expert: Expert, evaluated_rule: MultiClassTopRule):
        """
        Stop a conclusion by adding a stopping rule.

        :param case_query: The case query to stop the conclusion for.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param evaluated_rule: The evaluated rule to ask the expert about.
        """
        conditions = expert.ask_for_conditions(case_query, evaluated_rule)
        evaluated_rule.fit_rule(case_query.case, case_query.target, conditions=conditions)
        if self.mode == MCRDRMode.StopPlusRule:
            self.stop_rule_conditions = conditions
        if self.mode == MCRDRMode.StopPlusRuleCombined:
            new_top_rule_conditions = conditions.combine_with(evaluated_rule.conditions)
            self.add_top_rule(new_top_rule_conditions, case_query.target, case_query.case)

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
        return conclusion.__class__ == target.__class__ and target.__class__ != CaseAttribute

    def conclusion_is_correct(self, case_query: CaseQuery,
                              expert: Expert, evaluated_rule: Rule,
                              add_extra_conclusions: bool) -> bool:
        """
        Ask the expert if the conclusion is correct, and add it to the conclusions if it is.

        :param case_query: The case query to ask the expert about.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param evaluated_rule: The evaluated rule to ask the expert about.
        :param add_extra_conclusions: Whether adding extra conclusions after classification is allowed.
        :return: Whether the conclusion is correct or not.
        """
        conclusions = {case_query.attribute_name: c for c in OrderedSet(self.conclusions)}
        if (add_extra_conclusions and expert.ask_if_conclusion_is_correct(case_query.case,
                                                                          evaluated_rule.conclusion(case_query.case),
                                                                          targets=case_query.target(case_query.case),
                                                                          current_conclusions=conclusions)):
            self.add_conclusion(evaluated_rule, case_query.case)
            self.expert_accepted_conclusions.append(evaluated_rule.conclusion)
            return True
        return False

    def add_rule_for_case(self, case_query: CaseQuery, expert: Expert):
        """
        Add a rule for a case that has not been classified with any conclusion.

        :param case_query: The case query to add the rule for.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if self.stop_rule_conditions and self.mode == MCRDRMode.StopPlusRule:
            conditions = self.stop_rule_conditions
            self.stop_rule_conditions = None
        else:
            conditions = expert.ask_for_conditions(case_query)
        self.add_top_rule(conditions, case_query.target, case_query.case)

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

    def add_conclusion(self, evaluated_rule: Rule, case: Case) -> None:
        """
        Add the conclusion of the evaluated rule to the list of conclusions.

        :param evaluated_rule: The evaluated rule to add the conclusion of.
        :param case: The case to add the conclusion for.
        """
        conclusion_types = [type(c) for c in self.conclusions]
        rule_conclusion = evaluated_rule.conclusion(case)
        if type(rule_conclusion) not in conclusion_types:
            self.conclusions.extend(make_list(rule_conclusion))
        else:
            same_type_conclusions = [c for c in self.conclusions if type(c) == type(rule_conclusion)]
            combined_conclusion = rule_conclusion if isinstance(rule_conclusion, set) \
                else {rule_conclusion}
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

    def __init__(self, category_rdr_map: Optional[Dict[str, Union[SingleClassRDR, MultiClassRDR]]] = None):
        """
        :param category_rdr_map: A map of case attribute names to ripple down rules classifiers,
        where each category is a parent category that has a set of mutually exclusive (in case of SCRDR) child
        categories, e.g. {'species': SCRDR, 'habitats': MCRDR}, where 'species' and 'habitats' are attribute names
        for a case of type Animal, while SCRDR and MCRDR are SingleClass and MultiClass ripple down rules classifiers.
        Species can have values like Mammal, Bird, Fish, etc. which are mutually exclusive, while Habitat can have
        values like Land, Water, Air, etc., which are not mutually exclusive due to some animals living more than one
        habitat.
        """
        self.start_rules_dict: Dict[str, Union[SingleClassRDR, MultiClassRDR]] \
            = category_rdr_map if category_rdr_map else {}
        super(GeneralRDR, self).__init__()
        self.all_figs: List[plt.Figure] = [sr.fig for sr in self.start_rules_dict.values()]

    def add_rdr(self, rdr: Union[SingleClassRDR, MultiClassRDR], attribute_name: Optional[str] = None):
        """
        Add a ripple down rules classifier to the map of classifiers.

        :param rdr: The ripple down rules classifier to add.
        :param attribute_name: The name of the attribute that the classifier is classifying.
        """
        attribute_name = attribute_name if attribute_name else rdr.attribute_name
        self.start_rules_dict[attribute_name] = rdr

    @property
    def start_rule(self) -> Optional[Union[SingleClassRule, MultiClassTopRule]]:
        return self.start_rules[0] if self.start_rules_dict else None

    @start_rule.setter
    def start_rule(self, value: Union[SingleClassRDR, MultiClassRDR]):
        if value:
            self.start_rules_dict[value.attribute_name] = value

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
    def _classify(classifiers_dict: Dict[str, Union[ModuleType, RippleDownRules]],
                  case: Union[Case, SQLTable]) -> Optional[Dict[str, Any]]:
        """
        Classify a case by going through all classifiers and adding the categories that are classified,
         and then restarting the classification until no more categories can be added.

        :param classifiers_dict: A dictionary mapping conclusion types to the classifiers that produce them.
        :param case: The case to classify.
        :return: The categories that the case belongs to.
        """
        conclusions = {}
        case_cp = copy_case(case)
        while True:
            new_conclusions = {}
            for attribute_name, rdr in classifiers_dict.items():
                pred_atts = rdr.classify(case_cp)
                if pred_atts is None:
                    continue
                if rdr.type_ is SingleClassRDR:
                    if attribute_name not in conclusions or \
                            (attribute_name in conclusions and conclusions[attribute_name] != pred_atts):
                        conclusions[attribute_name] = pred_atts
                        new_conclusions[attribute_name] = pred_atts
                else:
                    pred_atts = make_set(pred_atts)
                    if attribute_name in conclusions:
                        pred_atts = {p for p in pred_atts if p not in conclusions[attribute_name]}
                    if len(pred_atts) > 0:
                        new_conclusions[attribute_name] = pred_atts
                        if attribute_name not in conclusions:
                            conclusions[attribute_name] = set()
                        conclusions[attribute_name].update(pred_atts)
                if attribute_name in new_conclusions:
                    mutually_exclusive = True if rdr.type_ is SingleClassRDR else False
                    GeneralRDR.update_case(CaseQuery(case_cp, attribute_name, rdr.conclusion_type, mutually_exclusive),
                                           new_conclusions)
            if len(new_conclusions) == 0:
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
        original_case_query_cp = copy(case_queries[0])
        for case_query in case_queries:
            case_query_cp = copy(case_query)
            case_query_cp.case = original_case_query_cp.case
            if case_query_cp.target is None:
                conclusions = self.classify(case) if self.start_rule and self.start_rule.conditions else []
                self.update_case(case_query_cp, conclusions)
                expert.ask_for_conclusion(case_query_cp)
                if case_query_cp.target is None:
                    continue
                case_query.target = case_query_cp.target

            if case_query.attribute_name not in self.start_rules_dict:
                conclusions = self.classify(case)
                self.update_case(case_query_cp, conclusions)

                new_rdr = self.initialize_new_rdr_for_attribute(case_query_cp)
                self.add_rdr(new_rdr, case_query.attribute_name)

                new_conclusions = new_rdr.fit_case(case_query_cp, expert, **kwargs)
                self.update_case(case_query_cp, {case_query.attribute_name: new_conclusions})
            else:
                for rdr_attribute_name, rdr in self.start_rules_dict.items():
                    if case_query.attribute_name != rdr_attribute_name:
                        conclusions = rdr.classify(case_query_cp.case)
                    else:
                        conclusions = self.start_rules_dict[rdr_attribute_name].fit_case(case_query_cp, expert,
                                                                                         **kwargs)
                    if conclusions is not None or (is_iterable(conclusions) and len(conclusions) > 0):
                        conclusions = {rdr_attribute_name: conclusions}
                        case_query_cp.mutually_exclusive = True if isinstance(rdr, SingleClassRDR) else False
                        self.update_case(case_query_cp, conclusions)
            case_query.conditions = case_query_cp.conditions

        return self.classify(case)

    @staticmethod
    def initialize_new_rdr_for_attribute(case_query: CaseQuery):
        """
        Initialize the appropriate RDR type for the target.
        """
        if case_query.mutually_exclusive is not None:
            return SingleClassRDR(default_conclusion=case_query.default_value) if case_query.mutually_exclusive\
                else MultiClassRDR()
        if case_query.attribute_type in [list, set]:
            return MultiClassRDR()
        attribute = getattr(case_query.case, case_query.attribute_name)\
            if hasattr(case_query.case, case_query.attribute_name) else case_query.target(case_query.case)
        if isinstance(attribute, CaseAttribute):
            return SingleClassRDR(default_conclusion=case_query.default_value) if attribute.mutually_exclusive \
                else MultiClassRDR()
        else:
            return MultiClassRDR() if is_iterable(attribute) or (attribute is None)\
                else SingleClassRDR(default_conclusion=case_query.default_value)

    @staticmethod
    def update_case(case_query: CaseQuery, conclusions: Dict[str, Any]):
        """
        Update the case with the conclusions.

        :param case_query: The case query that contains the case to update.
        :param conclusions: The conclusions to update the case with.
        """
        if not conclusions:
            return
        if len(conclusions) == 0:
            return
        if isinstance(case_query.original_case, SQLTable) or is_dataclass(case_query.original_case):
            for conclusion_name, conclusion in conclusions.items():
                attribute = getattr(case_query.case, conclusion_name)
                if conclusion_name == case_query.attribute_name:
                    attribute_type = case_query.attribute_type
                else:
                    attribute_type = (get_case_attribute_type(case_query.original_case, conclusion_name, attribute),)
                if isinstance(attribute, set) or any(at in {Set, set} for at in attribute_type):
                    attribute = set() if attribute is None else attribute
                    for c in conclusion:
                        attribute.update(make_set(c))
                elif isinstance(attribute, list) or any(at in {List, list} for at in attribute_type):
                    attribute = [] if attribute is None else attribute
                    attribute.extend(conclusion)
                elif is_iterable(conclusion) and len(conclusion) == 1 \
                        and any(at is type(list(conclusion)[0]) for at in attribute_type):
                    setattr(case_query.case, conclusion_name, list(conclusion)[0])
                elif not is_iterable(conclusion) and any(at is type(conclusion) for at in attribute_type):
                    setattr(case_query.case, conclusion_name, conclusion)
                else:
                    raise ValueError(f"Unknown type or type mismatch for attribute {conclusion_name} with type "
                                     f"{case_query.attribute_type} with conclusion "
                                     f"{conclusion} of type {type(conclusion)}")
        else:
            case_query.case.update(conclusions)

    def _to_json(self) -> Dict[str, Any]:
        return {"start_rules": {t: rdr.to_json() for t, rdr in self.start_rules_dict.items()}}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> GeneralRDR:
        """
        Create an instance of the class from a json
        """
        start_rules_dict = {}
        for k, v in data["start_rules"].items():
            start_rules_dict[k] = get_type_from_string(v['_type']).from_json(v)
        return cls(start_rules_dict)

    def write_to_python_file(self, file_path: str, postfix: str = "") -> None:
        """
        Write the tree of rules as source code to a file.

        :param file_path: The path to the file to write the source code to.
        :param postfix: The postfix to add to the file name.
        """
        self.generated_python_file_name = self._default_generated_python_file_name + postfix
        for rdr in self.start_rules_dict.values():
            rdr.write_to_python_file(file_path, postfix=f"_of_grdr{postfix}")
        func_def = f"def classify(case: {self.case_type.__name__}) -> {self.conclusion_type_hint}:\n"
        with open(file_path + f"/{self.generated_python_file_name}.py", "w") as f:
            f.write(self._get_imports(file_path) + "\n\n")
            f.write("classifiers_dict = dict()\n")
            for rdr_key, rdr in self.start_rules_dict.items():
                f.write(f"classifiers_dict['{rdr_key}'] = {self.rdr_key_to_function_name(rdr_key)}\n")
            f.write("\n\n")
            f.write(func_def)
            f.write(f"{' ' * 4}if not isinstance(case, Case):\n"
                    f"{' ' * 4}    case = create_case(case, max_recursion_idx=3)\n""")
            f.write(f"{' ' * 4}return GeneralRDR._classify(classifiers_dict, case)\n")

    @property
    def case_type(self) -> Type:
        """
        :return: The type of the case (input) to the RDR classifier.
        """
        if isinstance(self.start_rule.corner_case, Case):
            return self.start_rule.corner_case._obj_type
        else:
            return type(self.start_rule.corner_case)

    def get_rdr_classifier_from_python_file(self, file_path: str) -> Callable[[Any], Any]:
        """
        :param file_path: The path to the file that contains the RDR classifier function.
        :param postfix: The postfix to add to the file name.
        :return: The module that contains the rdr classifier function.
        """
        return importlib.import_module(f"{file_path.strip('./')}.{self.generated_python_file_name}").classify

    @property
    def generated_python_file_name(self) -> str:
        if self._generated_python_file_name is None:
            self._generated_python_file_name = self._default_generated_python_file_name
        return self._generated_python_file_name

    @generated_python_file_name.setter
    def generated_python_file_name(self, value: str):
        self._generated_python_file_name = value

    @property
    def _default_generated_python_file_name(self) -> str:
        """
        :return: The default generated python file name.
        """
        return f"{self.start_rule.corner_case._name.lower()}_rdr"

    @property
    def conclusion_type_hint(self) -> str:
        return f"List[Union[{', '.join([rdr.conclusion_type_hint for rdr in self.start_rules_dict.values()])}]]"

    def _get_imports(self, file_path: str) -> str:
        """
        Get the imports needed for the generated python file.

        :param file_path: The path to the file that contains the RDR classifier function.
        :return: The imports needed for the generated python file.
        """
        imports = ""
        # add type hints
        imports += f"from typing_extensions import List, Union, Set\n"
        # import rdr type
        imports += f"from ripple_down_rules.rdr import GeneralRDR\n"
        # add case type
        imports += f"from ripple_down_rules.datastructures.case import Case, create_case\n"
        imports += f"from {self.case_type.__module__} import {self.case_type.__name__}\n"
        # add conclusion type imports
        for rdr in self.start_rules_dict.values():
            for conclusion_type in rdr.conclusion_type:
                if conclusion_type.__module__ != "builtins":
                    imports += f"from {conclusion_type.__module__} import {conclusion_type.__name__}\n"
        # add rdr python generated functions.
        for rdr_key, rdr in self.start_rules_dict.items():
            imports += (f"from {file_path.strip('./')}"
                        f" import {rdr.generated_python_file_name} as {self.rdr_key_to_function_name(rdr_key)}\n")
        return imports

    @staticmethod
    def rdr_key_to_function_name(rdr_key: str) -> str:
        """
        Convert the RDR key to a function name.

        :param rdr_key: The RDR key to convert.
        :return: The function name.
        """
        return rdr_key.replace(".", "_").lower() + "_classifier"
