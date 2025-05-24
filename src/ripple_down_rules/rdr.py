from __future__ import annotations

import copyreg
import importlib
import sys
from abc import ABC, abstractmethod
from copy import copy
from io import TextIOWrapper
from types import ModuleType

from matplotlib import pyplot as plt
from sqlalchemy.orm import DeclarativeBase as SQLTable
from typing_extensions import List, Optional, Dict, Type, Union, Any, Self, Tuple, Callable, Set

from .datastructures.callable_expression import CallableExpression
from .datastructures.case import Case, CaseAttribute, create_case
from .datastructures.dataclasses import CaseQuery
from .datastructures.enums import MCRDRMode
from .experts import Expert, Human
from .helpers import is_matching
from .rules import Rule, SingleClassRule, MultiClassTopRule, MultiClassStopRule
from .user_interface.gui import RDRCaseViewer
from .utils import draw_tree, make_set, copy_case, \
    SubclassJSONSerializer, make_list, get_type_from_string, \
    is_conflicting, update_case, get_imports_from_scope, extract_function_source, extract_imports, get_full_class_name


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
    name: Optional[str] = None
    """
    The name of the classifier.
    """
    case_type: Optional[Type] = None
    """
    The type of the case (input) to the RDR classifier.
    """

    def __init__(self, start_rule: Optional[Rule] = None, viewer: Optional[RDRCaseViewer] = None):
        """
        :param start_rule: The starting rule for the classifier.
        """
        self.start_rule = start_rule
        self.fig: Optional[plt.Figure] = None
        self.viewer: Optional[RDRCaseViewer] = viewer
        if self.viewer is not None:
            self.viewer.set_save_function(self.save)

    def set_viewer(self, viewer: RDRCaseViewer):
        """
        Set the viewer for the classifier.

        :param viewer: The viewer to set.
        """
        self.viewer = viewer
        if self.viewer is not None:
            self.viewer.set_save_function(self.save)

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

    def __call__(self, case: Union[Case, SQLTable]) -> Union[CallableExpression, Dict[str, CallableExpression]]:
        return self.classify(case)

    @abstractmethod
    def classify(self, case: Union[Case, SQLTable], modify_case: bool = False) \
            -> Optional[Union[CallableExpression, Dict[str, CallableExpression]]]:
        """
        Classify a case.

        :param case: The case to classify.
        :param modify_case: Whether to modify the original case attributes with the conclusion or not.
        :return: The category that the case belongs to.
        """
        pass

    def fit_case(self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs) \
            -> Union[CallableExpression, Dict[str, CallableExpression]]:
        """
        Fit the classifier to a case and ask the expert for refinements or alternatives if the classification is
        incorrect by comparing the case with the target category.

        :param case_query: The query containing the case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        if case_query is None:
            raise ValueError("The case query cannot be None.")
        self.name = case_query.attribute_name if self.name is None else self.name
        self.case_type = case_query.case_type if self.case_type is None else self.case_type
        if case_query.target is None:
            case_query_cp = copy(case_query)
            self.classify(case_query_cp.case, modify_case=True)
            expert.ask_for_conclusion(case_query_cp)
            case_query.target = case_query_cp.target
            if case_query.target is None:
                return self.classify(case_query.case)

        self.update_start_rule(case_query, expert)

        return self._fit_case(case_query, expert=expert, **kwargs)

    @abstractmethod
    def _fit_case(self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs) \
            -> Union[CallableExpression, Dict[str, CallableExpression]]:
        """
        Fit the RDR on a case, and ask the expert for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category.

        :param case_query: The query containing the case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        pass

    @abstractmethod
    def update_start_rule(self, case_query: CaseQuery, expert: Expert):
        """
        Update the starting rule of the classifier.

        :param case_query: The case query to update the starting rule with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        pass

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

    @property
    def type_(self):
        return self.__class__

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
    @abstractmethod
    def _default_generated_python_file_name(self) -> str:
        """
        :return: The default generated python file name.
        """
        pass

    @abstractmethod
    def update_from_python_file(self, package_dir: str):
        """
        Update the rules from the generated python file, that might have been modified by the user.

        :param package_dir: The directory of the package that contains the generated python file.
        """
        pass


class RDRWithCodeWriter(RippleDownRules, ABC):

    def update_from_python_file(self, package_dir: str):
        """
        Update the rules from the generated python file, that might have been modified by the user.

        :param package_dir: The directory of the package that contains the generated python file.
        """
        rule_ids = [r.uid for r in [self.start_rule] + list(self.start_rule.descendants) if r.conditions is not None]
        condition_func_names = [f'conditions_{rid}' for rid in rule_ids]
        conclusion_func_names = [f'conclusion_{rid}' for rid in rule_ids]
        all_func_names = condition_func_names + conclusion_func_names
        filepath = f"{package_dir}/{self.generated_python_defs_file_name}.py"
        functions_source = extract_function_source(filepath, all_func_names, include_signature=False)
        # get the scope from the imports in the file
        scope = extract_imports(filepath)
        for rule in [self.start_rule] + list(self.start_rule.descendants):
            if rule.conditions is not None:
                rule.conditions.user_input = functions_source[f"conditions_{rule.uid}"]
                rule.conditions.scope = scope
            if rule.conclusion is not None:
                rule.conclusion.user_input = functions_source[f"conclusion_{rule.uid}"]
                rule.conclusion.scope = scope

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
        imports, defs_imports = self._get_imports()
        # clear the files first
        with open(defs_file_name, "w") as f:
            f.write(defs_imports + "\n\n")
        with open(file_name, "w") as f:
            imports += f"from .{self.generated_python_defs_file_name} import *\n"
            imports += f"from ripple_down_rules.rdr import {self.__class__.__name__}\n"
            f.write(imports + "\n\n")
            f.write(f"attribute_name = '{self.attribute_name}'\n")
            f.write(f"conclusion_type = ({', '.join([ct.__name__ for ct in self.conclusion_type])},)\n")
            f.write(f"type_ = {self.__class__.__name__}\n")
            f.write(f"\n\n{func_def}")
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

    def _get_imports(self) -> Tuple[str, str]:
        """
        :return: The imports for the generated python file of the RDR as a string.
        """
        defs_imports_list = []
        for rule in [self.start_rule] + list(self.start_rule.descendants):
            if not rule.conditions:
                continue
            for scope in [rule.conditions.scope, rule.conclusion.scope]:
                if scope is None:
                    continue
                defs_imports_list.extend(get_imports_from_scope(scope))
        if self.case_type.__module__ != "builtins":
            defs_imports_list.append(f"from {self.case_type.__module__} import {self.case_type.__name__}")
        defs_imports = "\n".join(set(defs_imports_list)) + "\n"
        imports = []
        if self.case_type.__module__ != "builtins":
            imports.append(f"from {self.case_type.__module__} import {self.case_type.__name__}")
        for conclusion_type in self.conclusion_type:
            if conclusion_type.__module__ != "builtins":
                imports.append(f"from {conclusion_type.__module__} import {conclusion_type.__name__}")
        imports.append("from ripple_down_rules.datastructures.case import Case, create_case")
        imports = set(imports).difference(defs_imports_list)
        imports = "\n".join(imports) + "\n"
        return imports, defs_imports

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
    def _default_generated_python_file_name(self) -> str:
        """
        :return: The default generated python file name.
        """
        if isinstance(self.start_rule.corner_case, Case):
            name = self.start_rule.corner_case._name
        else:
            name = self.start_rule.corner_case.__class__.__name__
        return f"{name.lower()}_{self.attribute_name}_{self.acronym.lower()}"

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
    def conclusion_type(self) -> Tuple[Type]:
        """
        :return: The type of the conclusion of the RDR classifier.
        """
        all_types = []
        for rule in [self.start_rule] + list(self.start_rule.descendants):
            all_types.extend(list(rule.conclusion.conclusion_type))
        return tuple(set(all_types))

    @property
    def attribute_name(self) -> str:
        """
        :return: The name of the attribute that the classifier is classifying.
        """
        return self.start_rule.conclusion_name

    def _to_json(self) -> Dict[str, Any]:
        return {"start_rule": self.start_rule.to_json(),  "generated_python_file_name": self.generated_python_file_name,
                "name": self.name, "case_type": get_full_class_name(self.case_type) if self.case_type is not None else None}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create an instance of the class from a json
        """
        start_rule = cls.start_rule_type().from_json(data["start_rule"])
        new_rdr = cls(start_rule)
        if "generated_python_file_name" in data:
            new_rdr.generated_python_file_name = data["generated_python_file_name"]
        if "name" in data:
            new_rdr.name = data["name"]
        if "case_type" in data:
            new_rdr.case_type = get_type_from_string(data["case_type"])
        return new_rdr

    @staticmethod
    @abstractmethod
    def start_rule_type() -> Type[Rule]:
        """
        :return: The type of the starting rule of the RDR classifier.
        """
        pass


class SingleClassRDR(RDRWithCodeWriter):

    def __init__(self, start_rule: Optional[SingleClassRule] = None, default_conclusion: Optional[Any] = None):
        """
        :param start_rule: The starting rule for the classifier.
        :param default_conclusion: The default conclusion for the classifier if no rules fire.
        """
        super(SingleClassRDR, self).__init__(start_rule)
        self.default_conclusion: Optional[Any] = default_conclusion

    def _fit_case(self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs) \
            -> Union[CaseAttribute, CallableExpression, None]:
        """
        Classify a case, and ask the user for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category if provided.

        :param case_query: The case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        if case_query.default_value is not None and self.default_conclusion != case_query.default_value:
            self.default_conclusion = case_query.default_value

        pred = self.evaluate(case_query.case)
        if pred.conclusion(case_query.case) != case_query.target_value:
            expert.ask_for_conditions(case_query, pred)
            pred.fit_rule(case_query.case, case_query.target, conditions=case_query.conditions)

        return self.classify(case_query.case)

    def update_start_rule(self, case_query: CaseQuery, expert: Expert):
        """
        Update the starting rule of the classifier.

        :param case_query: The case query to update the starting rule with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if not self.start_rule:
            expert.ask_for_conditions(case_query)
            self.start_rule = SingleClassRule(case_query.conditions, case_query.target, corner_case=case_query.case,
                                              conclusion_name=case_query.attribute_name)

    def classify(self, case: Case, modify_case: bool = False) -> Optional[Any]:
        """
        Classify a case by recursively evaluating the rules until a rule fires or the last rule is reached.

        :param case: The case to classify.
        :param modify_case: Whether to modify the original case attributes with the conclusion or not.
        """
        pred = self.evaluate(case)
        return pred.conclusion(case) if pred is not None and pred.fired else self.default_conclusion

    def evaluate(self, case: Case) -> SingleClassRule:
        """
        Evaluate the starting rule on a case.
        """
        matched_rule = self.start_rule(case) if self.start_rule is not None else None
        return matched_rule if matched_rule is not None else self.start_rule

    def write_to_python_file(self, file_path: str, postfix: str = ""):
        super().write_to_python_file(file_path, postfix)
        if self.default_conclusion is not None:
            with open(file_path + f"/{self.generated_python_file_name}.py", "a") as f:
                f.write(f"{' ' * 4}else:\n{' ' * 8}return {self.default_conclusion}\n")

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

            conclusion_call = rule.write_conclusion_as_source_code(parent_indent, defs_file)
            file.write(conclusion_call)

            if rule.alternative:
                self.write_rules_as_source_code_to_file(rule.alternative, file, parent_indent, defs_file=defs_file)

    @property
    def conclusion_type_hint(self) -> str:
        return self.conclusion_type[0].__name__

    @property
    def conclusion_type(self) -> Tuple[Type]:
        if self.default_conclusion is not None:
            return (type(self.default_conclusion),)
        return super().conclusion_type

    @staticmethod
    def start_rule_type() -> Type[Rule]:
        """
        :return: The type of the starting rule of the RDR classifier.
        """
        return SingleClassRule


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

    def __init__(self, start_rule: Optional[MultiClassTopRule] = None,
                 mode: MCRDRMode = MCRDRMode.StopOnly):
        """
        :param start_rule: The starting rules for the classifier.
        :param mode: The mode of the classifier, either StopOnly or StopPlusRule, or StopPlusRuleCombined.
        """
        super(MultiClassRDR, self).__init__(start_rule)
        self.mode: MCRDRMode = mode

    def classify(self, case: Union[Case, SQLTable], modify_case: bool = False) -> Set[Any]:
        evaluated_rule = self.start_rule
        self.conclusions = []
        while evaluated_rule:
            next_rule = evaluated_rule(case)
            if evaluated_rule.fired:
                self.add_conclusion(evaluated_rule, case)
            evaluated_rule = next_rule
        return make_set(self.conclusions)

    def _fit_case(self, case_query: CaseQuery, expert: Optional[Expert] = None
                  , **kwargs) -> Set[Union[CaseAttribute, CallableExpression, None]]:
        """
        Classify a case, and ask the user for stopping rules or classifying rules if the classification is incorrect
         or missing by comparing the case with the target category if provided.

        :param case_query: The query containing the case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions or for extra conclusions.
        :return: The conclusions that the case belongs to.
        """
        self.conclusions = []
        self.stop_rule_conditions = None
        evaluated_rule = self.start_rule
        target_value = make_set(case_query.target_value)
        while evaluated_rule:
            next_rule = evaluated_rule(case_query.case)
            rule_conclusion = evaluated_rule.conclusion(case_query.case)

            if evaluated_rule.fired:
                if not make_set(rule_conclusion).issubset(target_value):
                    # Rule fired and conclusion is different from target
                    self.stop_wrong_conclusion_else_add_it(case_query, expert, evaluated_rule)
                else:
                    # Rule fired and target is correct or there is no target to compare
                    self.add_conclusion(evaluated_rule, case_query.case)

            if not next_rule:
                if not make_set(target_value).issubset(make_set(self.conclusions)):
                    # Nothing fired and there is a target that should have been in the conclusions
                    self.add_rule_for_case(case_query, expert)
                    # Have to check all rules again to make sure only this new rule fires
                    next_rule = self.start_rule
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

            conclusion_call = rule.write_conclusion_as_source_code(conclusion_indent, defs_file)
            file.write(conclusion_call)

            if rule.alternative:
                self.write_rules_as_source_code_to_file(rule.alternative, file, parent_indent, defs_file=defs_file)

    @property
    def conclusion_type_hint(self) -> str:
        conclusion_types = [ct.__name__ for ct in self.conclusion_type if ct not in [list, set]]
        if len(conclusion_types) == 1:
            return f"Set[{conclusion_types[0]}]"
        else:
            return f"Set[Union[{', '.join(conclusion_types)}]]"

    def _get_imports(self) -> Tuple[str, str]:
        imports, defs_imports = super()._get_imports()
        imports += f"from typing_extensions import Set, Union\n"
        imports += "from ripple_down_rules.utils import make_set\n"
        defs_imports += "from typing_extensions import Union\n"
        return imports, defs_imports

    def update_start_rule(self, case_query: CaseQuery, expert: Expert):
        """
        Update the starting rule of the classifier.

        :param case_query: The case query to update the starting rule with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if not self.start_rule:
            conditions = expert.ask_for_conditions(case_query)
            self.start_rule = MultiClassTopRule(conditions, case_query.target, corner_case=case_query.case,
                                                conclusion_name=case_query.attribute_name)

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
                                          evaluated_rule: MultiClassTopRule):
        """
        Stop a wrong conclusion by adding a stopping rule.
        """
        rule_conclusion = evaluated_rule.conclusion(case_query.case)
        if is_conflicting(rule_conclusion, case_query.target_value):
            self.stop_conclusion(case_query, expert, evaluated_rule)
        else:
            self.add_conclusion(evaluated_rule, case_query.case)

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

    @staticmethod
    def start_rule_type() -> Type[Rule]:
        """
        :return: The type of the starting rule of the RDR classifier.
        """
        return MultiClassTopRule


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

    def __init__(self, category_rdr_map: Optional[Dict[str, Union[SingleClassRDR, MultiClassRDR]]] = None,
                 **kwargs):
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
        super(GeneralRDR, self).__init__(**kwargs)
        self.all_figs: List[plt.Figure] = [sr.fig for sr in self.start_rules_dict.values()]

    def add_rdr(self, rdr: Union[SingleClassRDR, MultiClassRDR], case_query: Optional[CaseQuery] = None):
        """
        Add a ripple down rules classifier to the map of classifiers.

        :param rdr: The ripple down rules classifier to add.
        :param case_query: The case query to add the classifier for.
        """
        name = case_query.attribute_name if case_query else rdr.name
        self.start_rules_dict[name] = rdr

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

    def classify(self, case: Any, modify_case: bool = False) -> Optional[Dict[str, Any]]:
        """
        Classify a case by going through all RDRs and adding the categories that are classified, and then restarting
        the classification until no more categories can be added.

        :param case: The case to classify.
        :param modify_case: Whether to modify the original case or create a copy and modify it.
        :return: The categories that the case belongs to.
        """
        return self._classify(self.start_rules_dict, case, modify_original_case=modify_case)

    @staticmethod
    def _classify(classifiers_dict: Dict[str, Union[ModuleType, RippleDownRules]],
                  case: Any, modify_original_case: bool = False) -> Dict[str, Any]:
        """
        Classify a case by going through all classifiers and adding the categories that are classified,
         and then restarting the classification until no more categories can be added.

        :param classifiers_dict: A dictionary mapping conclusion types to the classifiers that produce them.
        :param case: The case to classify.
        :param modify_original_case: Whether to modify the original case or create a copy and modify it.
        :return: The categories that the case belongs to.
        """
        conclusions = {}
        case = case if isinstance(case, (Case, SQLTable)) else create_case(case)
        case_cp = copy_case(case) if not modify_original_case else case
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
                    case_query = CaseQuery(case_cp, attribute_name, rdr.conclusion_type, mutually_exclusive)
                    update_case(case_query, new_conclusions)
            if len(new_conclusions) == 0:
                break
        return conclusions

    def _fit_case(self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs) \
            -> Dict[str, Any]:
        """
        Fit the GRDR on a case, if the target is a new type of category, a new RDR is created for it,
        else the existing RDR of that type will be fitted on the case, and then classification is done and all
        concluded categories are returned. If the category is mutually exclusive, an SCRDR is created, else an MCRDR.
        In case of SCRDR, multiple conclusions of the same type replace each other, in case of MCRDR, they are added if
        they are accepted by the expert, and the attribute of that category is represented in the case as a set of
        values.

        :param case_query: The query containing the case to classify and the target category to compare the case
        with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The categories that the case belongs to.
        """
        case_query_cp = copy(case_query)
        self.classify(case_query_cp.case, modify_case=True)
        case_query_cp.update_target_value()

        self.start_rules_dict[case_query_cp.attribute_name].fit_case(case_query_cp, expert, **kwargs)

        return self.classify(case_query.case)

    def update_start_rule(self, case_query: CaseQuery, expert: Expert):
        """
        Update the starting rule of the classifier.

        :param case_query: The case query to update the starting rule with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if case_query.attribute_name not in self.start_rules_dict:
            new_rdr = self.initialize_new_rdr_for_attribute(case_query)
            self.add_rdr(new_rdr, case_query)

    @staticmethod
    def initialize_new_rdr_for_attribute(case_query: CaseQuery):
        """
        Initialize the appropriate RDR type for the target.
        """
        return SingleClassRDR(default_conclusion=case_query.default_value) if case_query.mutually_exclusive \
            else MultiClassRDR()

    def _to_json(self) -> Dict[str, Any]:
        return {"start_rules": {name: rdr.to_json() for name, rdr in self.start_rules_dict.items()}
                , "generated_python_file_name": self.generated_python_file_name,
                "name": self.name,
                "case_type": get_full_class_name(self.case_type) if self.case_type is not None else None}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> GeneralRDR:
        """
        Create an instance of the class from a json
        """
        start_rules_dict = {}
        for k, v in data["start_rules"].items():
            start_rules_dict[k] = get_type_from_string(v['_type']).from_json(v)
        new_rdr = cls(start_rules_dict)
        if "generated_python_file_name" in data:
            new_rdr.generated_python_file_name = data["generated_python_file_name"]
        if "name" in data:
            new_rdr.name = data["name"]
        if "case_type" in data:
            new_rdr.case_type = get_type_from_string(data["case_type"])
        return new_rdr

    def update_from_python_file(self, package_dir: str) -> None:
        """
        Update the rules from the generated python file, that might have been modified by the user.

        :param package_dir: The directory of the package that contains the generated python file.
        """
        for rdr in self.start_rules_dict.values():
            rdr.update_from_python_file(package_dir)

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

    def get_rdr_classifier_from_python_file(self, file_path: str) -> Callable[[Any], Any]:
        """
        :param file_path: The path to the file that contains the RDR classifier function.
        :return: The module that contains the rdr classifier function.
        """
        return importlib.import_module(f"{file_path.strip('./')}.{self.generated_python_file_name}").classify

    @property
    def _default_generated_python_file_name(self) -> Optional[str]:
        """
        :return: The default generated python file name.
        """
        if self.start_rule is None or self.start_rule.conclusion is None:
            return None
        if isinstance(self.start_rule.corner_case, Case):
            name = self.start_rule.corner_case._name
        else:
            name = self.start_rule.corner_case.__class__.__name__
        return f"{name}_rdr".lower()

    @property
    def conclusion_type_hint(self) -> str:
        return f"Dict[str, Any]"

    def _get_imports(self, file_path: str) -> str:
        """
        Get the imports needed for the generated python file.

        :param file_path: The path to the file that contains the RDR classifier function.
        :return: The imports needed for the generated python file.
        """
        imports = ""
        # add type hints
        imports += f"from typing_extensions import Dict, Any\n"
        # import rdr type
        imports += f"from ripple_down_rules.rdr import GeneralRDR\n"
        # add case type
        imports += f"from ripple_down_rules.datastructures.case import Case, create_case\n"
        imports += f"from {self.case_type.__module__} import {self.case_type.__name__}\n"
        # add rdr python generated functions.
        for rdr_key, rdr in self.start_rules_dict.items():
            imports += (f"from ."
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
