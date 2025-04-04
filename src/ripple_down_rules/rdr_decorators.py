"""
This file contains decorators for the RDR (Ripple Down Rules) framework. Where each type of RDR has a decorator
that can be used with any python function such that this function can benefit from the incremental knowledge acquisition
of the RDRs.
"""

from functools import wraps
from typing import Callable, Optional, Type
from ripple_down_rules.datastructures import Case, Category, create_case, CaseQuery
from ripple_down_rules.experts import Expert, Human

from ripple_down_rules.rdr import SingleClassRDR, MultiClassRDR, GeneralRDR


def single_class_rdr(
    scrdr: SingleClassRDR,
    target: Optional[Type[Category]] = None,
    expert: Optional[Expert] = None
) -> Callable:
    """
    Decorator to use a SingleClassRDR as a classifier.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Category:
            func_arg_names = func.__code__.co_varnames
            func_arg_values = args + tuple(kwargs.values())
            case_dict = dict(zip(func_arg_names, func_arg_values))
            func_output = func(*args, **kwargs)
            case_dict.update({"_output": func_output})
            case = create_case(case_dict, recursion_idx=3)
            if target:
                return scrdr.fit_case(CaseQuery(case, target=target), expert=expert)
            else:
                return scrdr.classify(case)
        return wrapper

    return decorator
