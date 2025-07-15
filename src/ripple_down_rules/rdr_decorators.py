"""
This file contains decorators for the RDR (Ripple Down Rules) framework. Where each type of RDR has a decorator
that can be used with any python function such that this function can benefit from the incremental knowledge acquisition
of the RDRs.
"""
import os.path
from functools import wraps

from typing_extensions import Callable, Optional, Type, Tuple, Dict, Any, Self, get_type_hints, List, Union, Sequence

from .datastructures.case import Case
from .datastructures.dataclasses import CaseQuery, CaseFactoryMetaData
from .experts import Expert, Human
from .failures import RDRLoadError
from .rdr import GeneralRDR
from . import logger
try:
    from .user_interface.gui import RDRCaseViewer
except ImportError:
    RDRCaseViewer = None
from .utils import get_method_args_as_dict, get_func_rdr_model_name, make_set, \
    get_method_class_if_exists, str_to_snake_case
from .helpers import create_case_from_method


class RDRDecorator:
    rdr: GeneralRDR

    def __init__(self, models_dir: str,
                 output_type: Tuple[Type],
                 mutual_exclusive: bool,
                 output_name: str = "output_",
                 fit: bool = True,
                 expert: Optional[Expert] = None,
                 update_existing_rules: bool = True,
                 package_name: Optional[str] = None,
                 use_generated_classifier: bool = False,
                 ask_now: Callable[Dict[str, Any], bool] = lambda _: True,
                 fitting_decorator: Optional[Callable] = None,
                 generate_dot_file: bool = False) -> None:
        """
        :param models_dir: The directory to save/load the RDR models.
        :param output_type: The type of the output. This is used to create the RDR model.
        :param mutual_exclusive: If True, the output types are mutually exclusive.
         If None, the RDR model will not be saved as a python file.
        :param output_name: The name of the output. This is used to create the RDR model.
        :param fit: If True, the function will be in fit mode. This means that the RDR will prompt the user for the
            correct output if the function's output is not in the RDR model. If False, the function will be in
            classification mode. This means that the RDR will classify the function's output based on the RDR model.
        :param expert: The expert that will be used to prompt the user for the correct output. If None, a Human
            expert will be used.
        :param update_existing_rules: If True, the function will update the existing RDR rules
         even if they gave an output.
        :param package_name: The package name to use for relative imports in the RDR model.
        :param use_generated_classifier: If True, the function will use the generated classifier instead of the RDR model.
        :param ask_now: A callable that takes the case dictionary and returns True if the user should be asked for
            the output, or False if the function should return the output without asking.
        :param fitting_decorator: A decorator to use for the fitting function. If None, no decorator will be used.
        :param generate_dot_file: If True, the RDR model will generate a dot file for visualization.
        :return: A decorator to use a GeneralRDR as a classifier that monitors and modifies the function's output.
        """
        self.rdr_models_dir = models_dir
        self.rdr: Optional[GeneralRDR] = None
        self.model_name: Optional[str] = None
        self.output_type = output_type
        self.parsed_output_type: List[Type] = []
        self.mutual_exclusive = mutual_exclusive
        self.output_name = output_name
        self.fit: bool = fit
        self.expert: Optional[Expert] = expert
        self.update_existing_rules = update_existing_rules
        self.package_name = package_name
        self.use_generated_classifier = use_generated_classifier
        self.generated_classifier: Optional[Callable] = None
        self.ask_now = ask_now
        self.fitting_decorator = fitting_decorator if fitting_decorator is not None else \
            lambda f: f  # Default to no fitting decorator
        self.generate_dot_file = generate_dot_file
        self.not_none_output_found: bool = False
        # The following value will change dynamically each time the function is called.
        self.case_factory_metadata: CaseFactoryMetaData = CaseFactoryMetaData()
        self.load()

    def decorator(self, func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[Any]:

            if self.model_name is None:
                self.initialize_rdr_model_name_and_load(func)

            func_output = {self.output_name: func(*args, **kwargs)}

            case, case_dict = create_case_from_method(func, func_output, *args, **kwargs)

            @self.fitting_decorator
            def fit():
                if len(self.parsed_output_type) == 0:
                    self.parsed_output_type = self.parse_output_type(func, self.output_type, *args)
                if self.expert is None:
                    self.expert = Human(answers_save_path=self.rdr_models_dir + f'/{self.model_name}/expert_answers')
                case_query = self.create_case_query_from_method(
                                            func, func_output,
                                            self.parsed_output_type,
                                            self.mutual_exclusive,
                                            args, kwargs,
                                            case=case, case_dict=case_dict,
                                            scenario=self.case_factory_metadata.scenario,
                                            this_case_target_value=self.case_factory_metadata.this_case_target_value)
                output = self.rdr.fit_case(case_query, expert=self.expert,
                                           update_existing_rules=self.update_existing_rules)
                return output

            if self.fit and not self.use_generated_classifier and self.ask_now(case_dict):
                output = fit()
            else:
                if self.use_generated_classifier:
                    if self.generated_classifier is None:
                        model_path = os.path.join(self.rdr_models_dir, self.model_name)
                        self.generated_classifier = self.rdr.get_rdr_classifier_from_python_file(model_path)
                    output = self.generated_classifier(case)
                else:
                    output = self.rdr.classify(case)
                    if self.generate_dot_file:
                        eval_rule_tree = self.rdr.get_evaluated_rule_tree()
                        if not self.not_none_output_found or (eval_rule_tree and len(eval_rule_tree) > 1):
                            self.rdr.render_evaluated_rule_tree(self.rdr_models_dir + f'/{self.model_name}',
                                                                show_full_tree=True)
                        if eval_rule_tree and len(eval_rule_tree) > 1:
                            self.not_none_output_found = True

            if self.output_name in output:
                return output[self.output_name]
            else:
                return func_output[self.output_name]

        wrapper._rdr_decorator_instance = self

        return wrapper

    @staticmethod
    def create_case_query_from_method(func: Callable,
                                      func_output: Dict[str, Any],
                                      output_type: Sequence[Type],
                                      mutual_exclusive: bool,
                                      func_args: Tuple[Any, ...], func_kwargs: Dict[str, Any],
                                      case: Optional[Case] = None,
                                      case_dict: Optional[Dict[str, Any]] = None,
                                      scenario: Optional[Callable] = None,
                                      this_case_target_value: Optional[Any] = None,) -> CaseQuery:
        """
        Create a CaseQuery from the function and its arguments.

        :param func: The function to create a case from.
        :param func_output: The output of the function as a dictionary, where the key is the output name.
        :param output_type: The type of the output as a sequence of types.
        :param mutual_exclusive: If True, the output types are mutually exclusive.
        :param func_args: The positional arguments of the function.
        :param func_kwargs: The keyword arguments of the function.
        :param case: The case to create.
        :param case_dict: The dictionary of the case.
        :param scenario: The scenario that produced the given case.
        :param this_case_target_value: The target value for the case.
        :return: A CaseQuery object representing the case.
        """
        output_type = make_set(output_type)
        if case is None or case_dict is None:
            case, case_dict = create_case_from_method(func, func_output, *func_args, **func_kwargs)
        scope = func.__globals__
        scope.update(case_dict)
        func_args_type_hints = get_type_hints(func)
        output_name = list(func_output.keys())[0]
        func_args_type_hints.update({output_name: Union[tuple(output_type)]})
        return CaseQuery(case, output_name, tuple(output_type),
                         mutual_exclusive, scope=scope, scenario=scenario, this_case_target_value=this_case_target_value,
                         is_function=True, function_args_type_hints=func_args_type_hints)

    def initialize_rdr_model_name_and_load(self, func: Callable) -> None:
        self.model_name = get_func_rdr_model_name(func, include_file_name=True)
        self.load()

    @staticmethod
    def parse_output_type(func: Callable, output_type: Any, *args) -> List[Type]:
        parsed_output_type = []
        for ot in make_set(output_type):
            if ot is Self:
                func_class = get_method_class_if_exists(func, *args)
                if func_class is not None:
                    parsed_output_type.append(func_class)
                else:
                    raise ValueError(f"The function {func} is not a method of a class,"
                                     f" and the output type is {Self}.")
            else:
                parsed_output_type.append(ot)
        return parsed_output_type

    def save(self):
        """
        Save the RDR model to the specified directory.
        """
        self.rdr.save(self.rdr_models_dir, self.model_name, package_name=self.package_name)

    def load(self):
        """
        Load the RDR model from the specified directory.
        """
        self.rdr = None
        if self.model_name is not None:
            try:
                self.rdr = GeneralRDR.load(self.rdr_models_dir, self.model_name, package_name=self.package_name)
            except RDRLoadError as e:
                pass
        if self.rdr is None:
            self.rdr = GeneralRDR(save_dir=self.rdr_models_dir, model_name=self.model_name)

    def update_from_python(self):
        """
        Update the RDR model from a python file.
        """
        self.rdr.update_from_python(self.rdr_models_dir, package_name=self.package_name)


def fit_rdr_func(scenario: Callable, rdr_decorated_func: Callable, *func_args, **func_kwargs) -> None:
    rdr_decorated_func._rdr_decorator_instance.case_factory_metadata = CaseFactoryMetaData(scenario=scenario)
    rdr_decorated_func(*func_args, **func_kwargs)