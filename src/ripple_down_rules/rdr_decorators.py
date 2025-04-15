"""
This file contains decorators for the RDR (Ripple Down Rules) framework. Where each type of RDR has a decorator
that can be used with any python function such that this function can benefit from the incremental knowledge acquisition
of the RDRs.
"""
import os.path
from functools import wraps
from typing import Callable, Optional, Type

from sqlalchemy.orm import Session
from typing_extensions import Any

from ripple_down_rules.datastructures import Case, Category, create_case, CaseQuery
from ripple_down_rules.experts import Expert, Human

from ripple_down_rules.rdr import SingleClassRDR, MultiClassRDR, GeneralRDR
from ripple_down_rules.utils import get_method_args_as_dict, get_method_name, get_method_class_name_if_exists, \
    get_method_file_name, get_func_rdr_model_path


def single_class_rdr(
    model_dir: str,
    fit: bool = True,
    expert: Optional[Expert] = None,
    session: Optional[Session] = None,
) -> Callable:
    """
    Decorator to use a SingleClassRDR as a classifier.
    """
    expert = expert if expert else Human(session=session)

    def decorator(func: Callable) -> Callable:
        scrdr_model_path = get_func_rdr_model_path(func, model_dir)
        if os.path.exists(scrdr_model_path):
            scrdr = SingleClassRDR.load(scrdr_model_path)
            scrdr.session = session
        else:
            scrdr = SingleClassRDR(session=session)

        @wraps(func)
        def wrapper(*args, **kwargs) -> Category:
            case_dict = get_method_args_as_dict(func, *args, **kwargs)
            func_output = func(*args, **kwargs)
            if func_output is not None:
                case_dict.update({"_output": func_output})
            case = create_case(case_dict, recursion_idx=3)
            if fit:
                output = scrdr.fit_case(CaseQuery(case), expert=expert)
                scrdr.save(scrdr_model_path)
                return output
            else:
                return scrdr.classify(case)
        return wrapper

    return decorator
