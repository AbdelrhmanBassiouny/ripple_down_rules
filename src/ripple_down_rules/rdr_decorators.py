"""
This file contains decorators for the RDR (Ripple Down Rules) framework. Where each type of RDR has a decorator
that can be used with any python function such that this function can benefit from the incremental knowledge acquisition
of the RDRs.
"""
import os.path
from functools import wraps

from pyparsing.tools.cvt_pyparsing_pep8_names import camel_to_snake
from typing_extensions import Callable, Optional, Type, Tuple, Dict, Any, Self, get_type_hints, List, Union

from ripple_down_rules.datastructures.case import create_case, Case
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.datastructures.enums import Category
from ripple_down_rules.experts import Expert, Human
from ripple_down_rules.rdr import GeneralRDR, RippleDownRules
from ripple_down_rules.utils import get_method_args_as_dict, get_func_rdr_model_name, make_set, \
    get_method_class_if_exists, get_method_name, str_to_snake_case


class RDRDecorator:
    rdr: GeneralRDR

    def __init__(self, models_dir: str,
                 output_type: Tuple[Type],
                 mutual_exclusive: bool,
                 output_name: str = "output_",
                 fit: bool = True,
                 expert: Optional[Expert] = None):
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
        :return: A decorator to use a GeneralRDR as a classifier that monitors and modifies the function's output.
        """
        self.rdr_models_dir = models_dir
        self.model_name: Optional[str] = None
        self.output_type = output_type
        self.parsed_output_type: List[Type] = []
        self.mutual_exclusive = mutual_exclusive
        self.output_name = output_name
        self.fit: bool = fit
        self.expert: Optional[Expert] = expert
        self.load()

    def decorator(self, func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[Any]:

            if len(self.parsed_output_type) == 0:
                self.parsed_output_type = self.parse_output_type(func, self.output_type, *args)
            if self.model_name is None:
                self.initialize_rdr_model_name_and_load(func)

            if self.fit:
                case_query = self.create_case_query_from_method(func, self.parsed_output_type,
                                                                self.mutual_exclusive, self.output_name,
                                                                *args, **kwargs)
                output = self.rdr.fit_case(case_query, expert=self.expert)
                return output[self.output_name]
            else:
                case, case_dict = self.create_case_from_method(func, self.output_name, *args, **kwargs)
                return self.rdr.classify(case)[self.output_name]

        return wrapper

    @staticmethod
    def create_case_query_from_method(func: Callable, output_type, mutual_exclusive: bool,
                                output_name: str = 'output_', *args, **kwargs) -> CaseQuery:
        """
        Create a CaseQuery from the function and its arguments.

        :param func: The function to create a case from.
        :param output_type: The type of the output.
        :param mutual_exclusive: If True, the output types are mutually exclusive.
        :param output_name: The name of the output in the case. Defaults to 'output_'.
        :param args: The positional arguments of the function.
        :param kwargs: The keyword arguments of the function.
        :return: A CaseQuery object representing the case.
        """
        output_type = make_set(output_type)
        case, case_dict = RDRDecorator.create_case_from_method(func, output_name, *args, **kwargs)
        scope = func.__globals__
        scope.update(case_dict)
        func_args_type_hints = get_type_hints(func)
        func_args_type_hints.update({output_name: Union[tuple(output_type)]})
        return CaseQuery(case, output_name, Union[tuple(output_type)],
                         mutual_exclusive, scope=scope,
                         is_function=True, function_args_type_hints=func_args_type_hints)

    @staticmethod
    def create_case_from_method(func: Callable, output_name: str = "output_", *args, **kwargs) -> Tuple[Case, Dict[str, Any]]:
        """
        Create a Case from the function and its arguments.

        :param func: The function to create a case from.
        :param output_name: The name of the output in the case. Defaults to 'output_'.
        :param args: The positional arguments of the function.
        :param kwargs: The keyword arguments of the function.
        :return: A Case object representing the case.
        """
        case_dict = get_method_args_as_dict(func, *args, **kwargs)
        func_output = func(*args, **kwargs)
        case_dict.update({output_name: func_output})
        case_name = get_func_rdr_model_name(func)
        return create_case(case_dict, obj_name=case_name, max_recursion_idx=3), case_dict

    def initialize_rdr_model_name_and_load(self, func: Callable) -> None:
        model_file_name = get_func_rdr_model_name(func, include_file_name=True)
        self.model_name = str_to_snake_case(model_file_name)
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
        self.rdr.save(self.rdr_models_dir)

    def load(self):
        """
        Load the RDR model from the specified directory.
        """
        self.rdr = None
        if self.model_name is not None:
            model_path = os.path.join(self.rdr_models_dir, self.model_name + f"/rdr_metadata/{self.model_name}.json")
            if os.path.exists(os.path.join(self.rdr_models_dir, model_path)):
                self.rdr = GeneralRDR.load(self.rdr_models_dir, self.model_name)
        if self.rdr is None:
            self.rdr = GeneralRDR(save_dir=self.rdr_models_dir, model_name=self.model_name)

    def update_from_python(self):
        """
        Update the RDR model from a python file.
        """
        self.rdr.update_from_python(self.rdr_models_dir, self.model_name)
