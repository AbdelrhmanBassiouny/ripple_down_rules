from __future__ import annotations

import copyreg
import importlib
import os

from ripple_down_rules.datastructures.dataclasses import CaseFactoryMetaData

from . import logger
import sys
from abc import ABC, abstractmethod
from copy import copy
from io import TextIOWrapper
from types import ModuleType

try:
    from matplotlib import pyplot as plt
    Figure = plt.Figure
except ImportError as e:
    logger.debug(f"{e}: matplotlib is not installed")
    matplotlib = None
    Figure = None
    plt = None

from sqlalchemy.orm import DeclarativeBase as SQLTable
from typing_extensions import List, Optional, Dict, Type, Union, Any, Self, Tuple, Callable, Set

from .datastructures.callable_expression import CallableExpression
from .datastructures.case import Case, CaseAttribute, create_case
from .datastructures.dataclasses import CaseQuery
from .datastructures.enums import MCRDRMode
from .experts import Expert, Human
from .helpers import is_matching, general_rdr_classify
from .rules import Rule, SingleClassRule, MultiClassTopRule, MultiClassStopRule
try:
    from .user_interface.gui import RDRCaseViewer
except ImportError as e:
    RDRCaseViewer = None
from .utils import draw_tree, make_set, copy_case, \
    SubclassJSONSerializer, make_list, get_type_from_string, \
    is_conflicting, get_imports_from_scope, extract_function_source, extract_imports, get_full_class_name, \
    is_iterable, str_to_snake_case, get_import_path_from_path


class RippleDownRules(SubclassJSONSerializer, ABC):
    """
    The abstract base class for the ripple down rules classifiers.
    """
    fig: Optional[Figure] = None
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
    case_name: Optional[str] = None
    """
    The name of the case type.
    """
    metadata_folder: str = "rdr_metadata"
    """
    The folder to save the metadata of the RDR classifier.
    """
    model_name: Optional[str] = None
    """
    The name of the model. If None, the model name will be the generated python file name.
    """
    mutually_exclusive: Optional[bool] = None
    """
    Whether the output of the classification of this rdr allows only one possible conclusion or not.
    """

    def __init__(self, start_rule: Optional[Rule] = None, viewer: Optional[RDRCaseViewer] = None,
                 save_dir: Optional[str] = None, model_name: Optional[str] = None):
        """
        :param start_rule: The starting rule for the classifier.
        :param viewer: The viewer gui to use for the classifier. If None, no viewer is used.
        :param save_dir: The directory to save the classifier to.
        """
        self.model_name: Optional[str] = model_name
        self.save_dir = save_dir
        self.start_rule = start_rule
        self.fig: Optional[Figure] = None
        self.viewer: Optional[RDRCaseViewer] = viewer
        if self.viewer is not None:
            self.viewer.set_save_function(self.save)

    def save(self, save_dir: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """
        Save the classifier to a file.

        :param save_dir: The directory to save the classifier to.
        :param model_name: The name of the model to save. If None, a default name is generated.
        :param postfix: The postfix to add to the file name.
        :return: The name of the saved model.
        """
        save_dir = save_dir or self.save_dir
        if save_dir is None:
            raise ValueError("The save directory cannot be None. Please provide a valid directory to save"
                             " the classifier.")
        if not os.path.exists(save_dir + '/__init__.py'):
            os.makedirs(save_dir, exist_ok=True)
            with open(save_dir + '/__init__.py', 'w') as f:
                f.write("from . import *\n")
        if model_name is not None:
            self.model_name = model_name
        elif self.model_name is None:
            self.model_name = self.generated_python_file_name
        model_dir = os.path.join(save_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        json_dir = os.path.join(model_dir, self.metadata_folder)
        os.makedirs(json_dir, exist_ok=True)
        self.to_json_file(os.path.join(json_dir, self.model_name))
        self._write_to_python(model_dir)
        return self.model_name

    @classmethod
    def load(cls, load_dir: str, model_name: str) -> Self:
        """
        Load the classifier from a file.

        :param load_dir: The path to the model directory to load the classifier from.
        :param model_name: The name of the model to load.
        """
        model_dir = os.path.join(load_dir, model_name)
        json_file = os.path.join(model_dir, cls.metadata_folder, model_name)
        rdr = cls.from_json_file(json_file)
        try:
            rdr.update_from_python(model_dir)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not load the python file for the model {model_name} from {model_dir}. "
                           f"Make sure the file exists and is valid.")
        rdr.save_dir = load_dir
        rdr.model_name = model_name
        return rdr

    @abstractmethod
    def _write_to_python(self, model_dir: str):
        """
        Write the tree of rules as source code to a file.

        :param model_dir: The path to the directory to write the source code to.
        """
        pass

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
            if plt is None:
                raise ImportError("matplotlib is not installed, cannot animate the tree.")
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
                if animate_tree and len(self.start_rule.descendants) > num_rules:
                    num_rules = len(self.start_rule.descendants)
                    self.update_figures()
            i += 1
            all_predictions = [1 if is_matching(self.classify, case_query) else 0 for case_query in case_queries
                               if case_query.target is not None]
            all_pred = sum(all_predictions)
            logger.info(f"Accuracy: {all_pred}/{len(targets)}")
            all_predicted = targets and all_pred == len(targets)
            num_iter_reached = n_iter and i >= n_iter
            stop_iterating = all_predicted or num_iter_reached
            if stop_iterating:
                break
        logger.info(f"Finished training in {i} iterations")
        if animate_tree:
            plt.ioff()
            plt.show()

    def __call__(self, case: Union[Case, SQLTable]) -> Union[CallableExpression, Dict[str, CallableExpression]]:
        return self.classify(case)

    @abstractmethod
    def classify(self, case: Union[Case, SQLTable], modify_case: bool = False,
                 case_query: Optional[CaseQuery] = None) \
            -> Optional[Union[CallableExpression, Dict[str, CallableExpression]]]:
        """
        Classify a case.

        :param case: The case to classify.
        :param modify_case: Whether to modify the original case attributes with the conclusion or not.
        :param case_query: The case query containing the case to classify and the target category to compare the case with.
        :return: The category that the case belongs to.
        """
        pass

    def fit_case(self, case_query: CaseQuery,
                 expert: Optional[Expert] = None,
                 update_existing_rules: bool = True,
                 scenario: Optional[Callable] = None,
                 **kwargs) \
            -> Union[CallableExpression, Dict[str, CallableExpression]]:
        """
        Fit the classifier to a case and ask the expert for refinements or alternatives if the classification is
        incorrect by comparing the case with the target category.

        :param case_query: The query containing the case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param update_existing_rules: Whether to update the existing same conclusion type rules that already gave
        some conclusions with the type required by the case query.
        :param scenario: The scenario at which the case was created, this is used to recreate the case if needed.
        :return: The category that the case belongs to.
        """
        if case_query is None:
            raise ValueError("The case query cannot be None.")

        self.name = case_query.attribute_name if self.name is None else self.name
        self.case_type = case_query.case_type if self.case_type is None else self.case_type
        self.case_name = case_query.case_name if self.case_name is None else self.case_name
        case_query.scenario = scenario if case_query.scenario is None else case_query.scenario

        expert = expert or Human(viewer=self.viewer,
                                 answers_save_path=self.save_dir + '/expert_answers'
                                 if self.save_dir else None)
        if case_query.target is None:
            case_query_cp = copy(case_query)
            conclusions = self.classify(case_query_cp.case, modify_case=True, case_query=case_query_cp)
            if self.should_i_ask_the_expert_for_a_target(conclusions, case_query_cp, update_existing_rules):
                expert.ask_for_conclusion(case_query_cp)
                case_query.target = case_query_cp.target
            if case_query.target is None:
                return self.classify(case_query.case)

        self.update_start_rule(case_query, expert)

        fit_case_result = self._fit_case(case_query, expert=expert, **kwargs)

        if self.save_dir is not None:
            self.save()
            expert.clear_answers()

        return fit_case_result

    @staticmethod
    def should_i_ask_the_expert_for_a_target(conclusions: Union[Any, Dict[str, Any]],
                                             case_query: CaseQuery,
                                             update_existing: bool) -> bool:
        """
        Determine if the rdr should ask the expert for the target of a given case query.

        :param conclusions: The conclusions of the case.
        :param case_query: The query containing the case to classify.
        :param update_existing: Whether to update rules that gave the required type of conclusions.
        :return: True if the rdr should ask the expert, False otherwise.
        """
        if conclusions is None:
            return True
        elif is_iterable(conclusions) and len(conclusions) == 0:
            return True
        elif isinstance(conclusions, dict):
            if case_query.attribute_name not in conclusions:
                return True
            conclusions = conclusions[case_query.attribute_name]
        conclusion_types = map(type, make_list(conclusions))
        if not any(ct in case_query.core_attribute_type for ct in conclusion_types):
            return True
        elif update_existing:
            return True
        else:
            return False

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
    def update_from_python(self, model_dir: str):
        """
        Update the rules from the generated python file, that might have been modified by the user.

        :param model_dir: The directory where the generated python file is located.
        """
        pass

    @classmethod
    def get_acronym(cls) -> str:
        """
        :return: The acronym of the classifier.
        """
        if cls.__name__ == "GeneralRDR":
            return "RDR"
        elif cls.__name__ == "MultiClassRDR":
            return "MCRDR"
        else:
            return "SCRDR"

    def get_rdr_classifier_from_python_file(self, package_name: str) -> Callable[[Any], Any]:
        """
        :param package_name: The name of the package that contains the RDR classifier function.
        :return: The module that contains the rdr classifier function.
        """
        # remove from imports if exists first
        package_name = get_import_path_from_path(package_name)
        name = f"{package_name}.{self.generated_python_file_name}" if package_name else self.generated_python_file_name
        try:
            module = importlib.import_module(name)
            del sys.modules[name]
        except ModuleNotFoundError:
            pass
        return importlib.import_module(name).classify


class RDRWithCodeWriter(RippleDownRules, ABC):

    def update_from_python(self, model_dir: str):
        """
        Update the rules from the generated python file, that might have been modified by the user.

        :param model_dir: The directory where the generated python file is located.
        """
        rules_dict = {r.uid: r for r in [self.start_rule] + list(self.start_rule.descendants) if r.conditions is not None}
        condition_func_names = [f'conditions_{rid}' for rid in rules_dict.keys()]
        conclusion_func_names = [f'conclusion_{rid}' for rid in rules_dict.keys() if not isinstance(rules_dict[rid], MultiClassStopRule)]
        all_func_names = condition_func_names + conclusion_func_names
        filepath = f"{model_dir}/{self.generated_python_defs_file_name}.py"
        cases_path = f"{model_dir}/{self.generated_python_cases_file_name}.py"
        cases_import_path = get_import_path_from_path(model_dir)
        cases_import_path = f"{cases_import_path}.{self.generated_python_cases_file_name}" if cases_import_path\
            else self.generated_python_cases_file_name
        functions_source = extract_function_source(filepath, all_func_names, include_signature=False)
        # get the scope from the imports in the file
        scope = extract_imports(filepath)
        for rule in [self.start_rule] + list(self.start_rule.descendants):
            if rule.conditions is not None:
                rule.conditions.user_input = functions_source[f"conditions_{rule.uid}"]
                rule.conditions.scope = scope
                if os.path.exists(cases_path):
                    rule.corner_case_metadata = importlib.import_module(cases_import_path).__dict__.get(f"corner_case_{rule.uid}", None)
            if rule.conclusion is not None and not isinstance(rule, MultiClassStopRule):
                rule.conclusion.user_input = functions_source[f"conclusion_{rule.uid}"]
                rule.conclusion.scope = scope

    @abstractmethod
    def write_rules_as_source_code_to_file(self, rule: Rule, file, parent_indent: str = "",
                                           defs_file: Optional[str] = None, cases_file: Optional[str] = None):
        """
        Write the rules as source code to a file.

        :param rule: The rule to write as source code.
        :param file: The file to write the source code to.
        :param parent_indent: The indentation of the parent rule.
        :param defs_file: The file to write the definitions to.
        :param cases_file: The file to write the cases to.
        """
        pass

    def _write_to_python(self, model_dir: str):
        """
        Write the tree of rules as source code to a file.

        :param model_dir: The path to the directory to write the source code to.
        """
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(model_dir + '/__init__.py'):
            with open(model_dir + '/__init__.py', 'w') as f:
                f.write("from . import *\n")
        func_def = f"def classify(case: {self.case_type.__name__}, **kwargs) -> {self.conclusion_type_hint}:\n"
        file_name = model_dir + f"/{self.generated_python_file_name}.py"
        defs_file_name = model_dir + f"/{self.generated_python_defs_file_name}.py"
        cases_file_name = model_dir + f"/{self.generated_python_cases_file_name}.py"
        imports, defs_imports = self._get_imports()
        # clear the files first
        with open(defs_file_name, "w") as f:
            f.write(defs_imports + "\n\n")
        with open(cases_file_name, "w") as cases_f:
            cases_f.write("# This file contains the corner cases for the rules.\n")
        with open(file_name, "w") as f:
            imports += f"from .{self.generated_python_defs_file_name} import *\n"
            f.write(imports + "\n\n")
            f.write(f"attribute_name = '{self.attribute_name}'\n")
            f.write(f"conclusion_type = ({', '.join([ct.__name__ for ct in self.conclusion_type])},)\n")
            f.write(f"mutually_exclusive = {self.mutually_exclusive}\n")
            f.write(f"\n\n{func_def}")
            f.write(f"{' ' * 4}if not isinstance(case, Case):\n"
                    f"{' ' * 4}    case = create_case(case, max_recursion_idx=3)\n""")
            self.write_rules_as_source_code_to_file(self.start_rule, f, " " * 4, defs_file=defs_file_name,
                                                    cases_file=cases_file_name)

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

    @property
    def _default_generated_python_file_name(self) -> Optional[str]:
        """
        :return: The default generated python file name.
        """
        if self.start_rule is None or self.start_rule.conclusion is None:
            return None
        return f"{str_to_snake_case(self.case_name)}_{self.attribute_name}_{self.get_acronym().lower()}"

    @property
    def generated_python_defs_file_name(self) -> str:
        return f"{self.generated_python_file_name}_defs"

    @property
    def generated_python_cases_file_name(self) -> str:
        return f"{self.generated_python_file_name}_cases"


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
        return {"start_rule": self.start_rule.to_json(),
                "generated_python_file_name": self.generated_python_file_name,
                "name": self.name,
                "case_type": get_full_class_name(self.case_type) if self.case_type is not None else None,
                "case_name": self.case_name}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create an instance of the class from a json
        """
        start_rule = cls.start_rule_type().from_json(data["start_rule"])
        new_rdr = cls(start_rule=start_rule)
        if "generated_python_file_name" in data:
            new_rdr.generated_python_file_name = data["generated_python_file_name"]
        if "name" in data:
            new_rdr.name = data["name"]
        if "case_type" in data:
            new_rdr.case_type = get_type_from_string(data["case_type"])
        if "case_name" in data:
            new_rdr.case_name = data["case_name"]
        return new_rdr

    @staticmethod
    @abstractmethod
    def start_rule_type() -> Type[Rule]:
        """
        :return: The type of the starting rule of the RDR classifier.
        """
        pass


class SingleClassRDR(RDRWithCodeWriter):

    mutually_exclusive: bool = True
    """
    The output of the classification of this rdr negates all other possible outputs, there can only be one true value.
    """

    def __init__(self, default_conclusion: Optional[Any] = None, **kwargs):
        """
        :param start_rule: The starting rule for the classifier.
        :param default_conclusion: The default conclusion for the classifier if no rules fire.
        """
        super(SingleClassRDR, self).__init__(**kwargs)
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
            pred.fit_rule(case_query)

        return self.classify(case_query.case)

    def update_start_rule(self, case_query: CaseQuery, expert: Expert):
        """
        Update the starting rule of the classifier.

        :param case_query: The case query to update the starting rule with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if not self.start_rule:
            expert.ask_for_conditions(case_query)
            self.start_rule = SingleClassRule.from_case_query(case_query)

    def classify(self, case: Case, modify_case: bool = False,
                 case_query: Optional[CaseQuery] = None) -> Optional[Any]:
        """
        Classify a case by recursively evaluating the rules until a rule fires or the last rule is reached.

        :param case: The case to classify.
        :param modify_case: Whether to modify the original case attributes with the conclusion or not.
        :param case_query: The case query containing the case and the target category to compare the case with.
        """
        pred = self.evaluate(case)
        conclusion = pred.conclusion(case) if pred is not None else None
        if pred is not None and pred.fired and case_query is not None:
            if pred.corner_case_metadata is None and conclusion is not None\
                    and type(conclusion) in case_query.core_attribute_type:
                pred.corner_case_metadata = CaseFactoryMetaData.from_case_query(case_query)
        return conclusion if pred is not None and pred.fired else self.default_conclusion

    def evaluate(self, case: Case) -> SingleClassRule:
        """
        Evaluate the starting rule on a case.
        """
        matched_rule = self.start_rule(case) if self.start_rule is not None else None
        return matched_rule if matched_rule is not None else self.start_rule

    def _write_to_python(self, model_dir: str):
        super()._write_to_python(model_dir)
        if self.default_conclusion is not None:
            with open(model_dir + f"/{self.generated_python_file_name}.py", "a") as f:
                f.write(f"{' ' * 4}else:\n{' ' * 8}return {self.default_conclusion}\n")

    def write_rules_as_source_code_to_file(self, rule: SingleClassRule, file: TextIOWrapper, parent_indent: str = "",
                                           defs_file: Optional[str] = None, cases_file: Optional[str] = None):
        """
        Write the rules as source code to a file.
        """
        if rule.conditions:
            rule.write_corner_case_as_source_code(cases_file)
            if_clause = rule.write_condition_as_source_code(parent_indent, defs_file)
            file.write(if_clause)
            if rule.refinement:
                self.write_rules_as_source_code_to_file(rule.refinement, file, parent_indent + "    ",
                                                        defs_file=defs_file, cases_file=cases_file)

            conclusion_call = rule.write_conclusion_as_source_code(parent_indent, defs_file)
            file.write(conclusion_call)

            if rule.alternative:
                self.write_rules_as_source_code_to_file(rule.alternative, file, parent_indent, defs_file=defs_file,
                                                        cases_file=cases_file)

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
    mutually_exclusive: bool = False
    """
    The output of the classification of this rdr allows for more than one true value as conclusion.
    """

    def __init__(self, start_rule: Optional[MultiClassTopRule] = None,
                 mode: MCRDRMode = MCRDRMode.StopOnly, **kwargs):
        """
        :param start_rule: The starting rules for the classifier.
        :param mode: The mode of the classifier, either StopOnly or StopPlusRule, or StopPlusRuleCombined.
        """
        super(MultiClassRDR, self).__init__(start_rule, **kwargs)
        self.mode: MCRDRMode = mode

    def classify(self, case: Union[Case, SQLTable], modify_case: bool = False,
                 case_query: Optional[CaseQuery] = None) -> Set[Any]:
        evaluated_rule = self.start_rule
        self.conclusions = []
        while evaluated_rule:
            next_rule = evaluated_rule(case)
            if evaluated_rule.fired:
                rule_conclusion = evaluated_rule.conclusion(case)
                if evaluated_rule.corner_case_metadata is None and case_query is not None:
                    if rule_conclusion is not None and len(make_list(rule_conclusion)) > 0\
                            and any(ct in case_query.core_attribute_type for ct in map(type, make_list(rule_conclusion))):
                        evaluated_rule.corner_case_metadata = CaseFactoryMetaData.from_case_query(case_query)
                self.add_conclusion(rule_conclusion)
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
                    self.add_conclusion(rule_conclusion)

            if not next_rule:
                if not make_set(target_value).issubset(make_set(self.conclusions)):
                    # Nothing fired and there is a target that should have been in the conclusions
                    self.add_rule_for_case(case_query, expert)
                    # Have to check all rules again to make sure only this new rule fires
                    next_rule = self.start_rule
            evaluated_rule = next_rule
        return self.conclusions

    def write_rules_as_source_code_to_file(self, rule: Union[MultiClassTopRule, MultiClassStopRule],
                                           file, parent_indent: str = "", defs_file: Optional[str] = None,
                                           cases_file: Optional[str] = None):
        if rule == self.start_rule:
            file.write(f"{parent_indent}conclusions = set()\n")
        if rule.conditions:
            rule.write_corner_case_as_source_code(cases_file)
            if_clause = rule.write_condition_as_source_code(parent_indent, defs_file)
            file.write(if_clause)
            conclusion_indent = parent_indent
            if hasattr(rule, "refinement") and rule.refinement:
                self.write_rules_as_source_code_to_file(rule.refinement, file, parent_indent + "    ",
                                                        defs_file=defs_file, cases_file=cases_file)
                conclusion_indent = parent_indent + " " * 4
                file.write(f"{conclusion_indent}else:\n")

            conclusion_call = rule.write_conclusion_as_source_code(conclusion_indent, defs_file)
            file.write(conclusion_call)

            if rule.alternative:
                self.write_rules_as_source_code_to_file(rule.alternative, file, parent_indent, defs_file=defs_file,
                                                        cases_file=cases_file)

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
            self.start_rule = MultiClassTopRule.from_case_query(case_query)

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
            self.add_conclusion(rule_conclusion)

    def stop_conclusion(self, case_query: CaseQuery,
                        expert: Expert, evaluated_rule: MultiClassTopRule):
        """
        Stop a conclusion by adding a stopping rule.

        :param case_query: The case query to stop the conclusion for.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param evaluated_rule: The evaluated rule to ask the expert about.
        """
        conditions = expert.ask_for_conditions(case_query, evaluated_rule)
        evaluated_rule.fit_rule(case_query)
        if self.mode == MCRDRMode.StopPlusRule:
            self.stop_rule_conditions = conditions
        if self.mode == MCRDRMode.StopPlusRuleCombined:
            new_top_rule_conditions = conditions.combine_with(evaluated_rule.conditions)
            case_query.conditions = new_top_rule_conditions
            self.add_top_rule(case_query)

    def add_rule_for_case(self, case_query: CaseQuery, expert: Expert):
        """
        Add a rule for a case that has not been classified with any conclusion.

        :param case_query: The case query to add the rule for.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if self.stop_rule_conditions and self.mode == MCRDRMode.StopPlusRule:
            conditions = self.stop_rule_conditions
            self.stop_rule_conditions = None
            case_query.conditions = conditions
        else:
            conditions = expert.ask_for_conditions(case_query)
        self.add_top_rule(case_query)

    def add_conclusion(self, rule_conclusion: List[Any]) -> None:
        """
        Add the conclusion of the evaluated rule to the list of conclusions.

        :param rule_conclusion: The conclusion of the evaluated rule, which can be a single conclusion
         or a set of conclusions.
        """
        conclusion_types = [type(c) for c in self.conclusions]
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

    def add_top_rule(self, case_query: CaseQuery):
        """
        Add a top rule to the classifier, which is a rule that is always checked and is part of the start_rules list.

        :param case_query: The case query to add the top rule for.
        """
        self.start_rule.alternative = MultiClassTopRule.from_case_query(case_query)

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
        self.all_figs: List[Figure] = [sr.fig for sr in self.start_rules_dict.values()]

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

    def classify(self, case: Any, modify_case: bool = False,
                 case_query: Optional[CaseQuery] = None) -> Optional[Dict[str, Any]]:
        """
        Classify a case by going through all RDRs and adding the categories that are classified, and then restarting
        the classification until no more categories can be added.

        :param case: The case to classify.
        :param modify_case: Whether to modify the original case or create a copy and modify it.
        :param case_query: The case query containing the case and the target category to compare the case with.
        :return: The categories that the case belongs to.
        """
        return general_rdr_classify(self.start_rules_dict, case, modify_original_case=modify_case,
                                    case_query=case_query)

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
                "case_type": get_full_class_name(self.case_type) if self.case_type is not None else None,
                "case_name": self.case_name}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> GeneralRDR:
        """
        Create an instance of the class from a json
        """
        start_rules_dict = {}
        for k, v in data["start_rules"].items():
            start_rules_dict[k] = get_type_from_string(v['_type']).from_json(v)
        new_rdr = cls(category_rdr_map=start_rules_dict)
        if "generated_python_file_name" in data:
            new_rdr.generated_python_file_name = data["generated_python_file_name"]
        if "name" in data:
            new_rdr.name = data["name"]
        if "case_type" in data:
            new_rdr.case_type = get_type_from_string(data["case_type"])
        if "case_name" in data:
            new_rdr.case_name = data["case_name"]
        return new_rdr

    def update_from_python(self, model_dir: str) -> None:
        """
        Update the rules from the generated python file, that might have been modified by the user.

        :param model_dir: The directory where the model is stored.
        """
        for rdr in self.start_rules_dict.values():
            rdr.update_from_python(model_dir)

    def _write_to_python(self, model_dir: str) -> None:
        """
        Write the tree of rules as source code to a file.

        :param model_dir: The directory where the model is stored.
        """
        for rdr in self.start_rules_dict.values():
            rdr._write_to_python(model_dir)
        func_def = f"def classify(case: {self.case_type.__name__}, **kwargs) -> {self.conclusion_type_hint}:\n"
        with open(model_dir + f"/{self.generated_python_file_name}.py", "w") as f:
            f.write(self._get_imports() + "\n\n")
            f.write("classifiers_dict = dict()\n")
            for rdr_key, rdr in self.start_rules_dict.items():
                f.write(f"classifiers_dict['{rdr_key}'] = {self.rdr_key_to_function_name(rdr_key)}\n")
            f.write("\n\n")
            f.write(func_def)
            f.write(f"{' ' * 4}if not isinstance(case, Case):\n"
                    f"{' ' * 4}    case = create_case(case, max_recursion_idx=3)\n""")
            f.write(f"{' ' * 4}return general_rdr_classify(classifiers_dict, case, **kwargs)\n")

    @property
    def _default_generated_python_file_name(self) -> Optional[str]:
        """
        :return: The default generated python file name.
        """
        if self.start_rule is None or self.start_rule.conclusion is None:
            return None
        return f"{str_to_snake_case(self.case_name)}_rdr".lower()

    @property
    def conclusion_type_hint(self) -> str:
        return f"Dict[str, Any]"

    def _get_imports(self) -> str:
        """
        Get the imports needed for the generated python file.

        :return: The imports needed for the generated python file.
        """
        imports = ""
        # add type hints
        imports += f"from typing_extensions import Dict, Any\n"
        # import rdr type
        imports += f"from ripple_down_rules.helpers import general_rdr_classify\n"
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
