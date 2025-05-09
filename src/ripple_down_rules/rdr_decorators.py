"""
This file contains decorators for the RDR (Ripple Down Rules) framework. Where each type of RDR has a decorator
that can be used with any python function such that this function can benefit from the incremental knowledge acquisition
of the RDRs.
"""
import os.path
from functools import wraps
from typing_extensions import Callable, Optional, Type, Tuple, Dict, Any

from ripple_down_rules.datastructures.case import create_case
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.datastructures.enums import Category
from ripple_down_rules.experts import Expert, Human
from ripple_down_rules.rdr import GeneralRDR, RippleDownRules
from ripple_down_rules.utils import get_method_args_as_dict, get_func_rdr_model_name


class RDRDecorator:
    rdr: GeneralRDR

    def __init__(self, models_dir: str,
                 output_type: Tuple[Type],
                 mutual_exclusive: bool,
                 output_name: str = "output",
                 fit: bool = True,
                 expert: Optional[Expert] = None):
        """
        :param models_dir: The directory to save/load the RDR models.
        :param output_type: The type of the output. This is used to create the RDR model.
        :param mutual_exclusive: If True, the output types are mutually exclusive.
        :param output_name: The name of the output. This is used to create the RDR model.
        :param fit: If True, the function will be in fit mode. This means that the RDR will prompt the user for the
            correct output if the function's output is not in the RDR model. If False, the function will be in
            classification mode. This means that the RDR will classify the function's output based on the RDR model.
        :param expert: The expert that will be used to prompt the user for the correct output. If None, a Human
            expert will be used.
        :return: A decorator to use a GeneralRDR as a classifier that monitors and modifies the function's output.
        """
        self.rdr_models_dir = models_dir
        self.output_type = output_type
        self.mutual_exclusive = mutual_exclusive
        self.output_name = output_name
        self.fit: bool = fit
        self.expert = expert if expert else Human()
        self.rdr_model_path: Optional[str] = None
        self.load()

    def decorator(self, func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[Any]:
            if self.rdr_model_path is None:
                model_file_name = get_func_rdr_model_name(func, include_file_name=True)
                model_file_name = (''.join(['_' + c.lower() if c.isupper() else c for c in model_file_name]).lstrip('_')
                                   .replace('__', '_') + ".json")
                self.rdr_model_path = os.path.join(self.rdr_models_dir, model_file_name)
                self.load()
            case_dict = get_method_args_as_dict(func, *args, **kwargs)
            func_output = func(*args, **kwargs)
            if func_output is not None:
                case_dict.update({self.output_name: func_output})
            case = create_case(case_dict, obj_name=get_func_rdr_model_name(func), max_recursion_idx=3)
            if self.fit:
                scope = func.__globals__
                scope.update(case_dict)
                case_query = CaseQuery(case, self.output_name, self.output_type, self.mutual_exclusive,
                                       scope=scope, is_function=True)
                output = self.rdr.fit_case(case_query, expert=self.expert)
                return output[self.output_name]
            else:
                return self.rdr.classify(case)

        return wrapper

    def save(self):
        """
        Save the RDR model to the specified directory.
        """
        self.rdr.save(self.rdr_model_path)

    def load(self):
        """
        Load the RDR model from the specified directory.
        """
        if self.rdr_model_path is not None and os.path.exists(self.rdr_model_path):
            self.rdr = GeneralRDR.load(self.rdr_model_path)
        else:
            self.rdr = GeneralRDR()

    def write_to_python_file(self, package_dir: str, file_name_postfix: str = ""):
        """
        Write the RDR model to a python file.

        :param package_dir: The path to the directory to write the python file.
        """
        self.rdr.write_to_python_file(package_dir, postfix=file_name_postfix)

    def update_from_python_file(self, package_dir: str):
        """
        Update the RDR model from a python file.

        :param package_dir: The directory of the package that contains the generated python file.
        """
        self.rdr.update_from_python_file(package_dir)
