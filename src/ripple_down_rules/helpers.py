from __future__ import annotations

import os
from types import ModuleType
from typing import Tuple

from typing_extensions import Type, Optional, Callable, Any, Dict, TYPE_CHECKING, Union

from .datastructures.case import create_case, Case
from .datastructures.dataclasses import CaseQuery
from .utils import calculate_precision_and_recall
from .utils import get_func_rdr_model_name, copy_case, make_set, update_case

if TYPE_CHECKING:
    from .rdr import RippleDownRules


def general_rdr_classify(classifiers_dict: Dict[str, Union[ModuleType, RippleDownRules]],
                         case: Any, modify_original_case: bool = False,
                         case_query: Optional[CaseQuery] = None) -> Dict[str, Any]:
    """
    Classify a case by going through all classifiers and adding the categories that are classified,
     and then restarting the classification until no more categories can be added.

    :param classifiers_dict: A dictionary mapping conclusion types to the classifiers that produce them.
    :param case: The case to classify.
    :param modify_original_case: Whether to modify the original case or create a copy and modify it.
    :param case_query: The case query to extract metadata from if needed.
    :return: The categories that the case belongs to.
    """
    conclusions = {}
    case = create_case(case)
    case_cp = copy_case(case) if not modify_original_case else case
    while True:
        new_conclusions = {}
        for attribute_name, rdr in classifiers_dict.items():
            pred_atts = rdr.classify(case_cp, case_query=case_query)
            if pred_atts is None and type(None) not in rdr.conclusion_type:
                continue
            if rdr.mutually_exclusive:
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
                temp_case_query = CaseQuery(case_cp, attribute_name, rdr.conclusion_type, rdr.mutually_exclusive)
                update_case(temp_case_query, new_conclusions)
        if len(new_conclusions) == 0 or len(classifiers_dict) == 1 and list(classifiers_dict.values())[
            0].mutually_exclusive:
            break
    return conclusions


def is_matching(classifier: Callable[[Any], Any], case_query: CaseQuery,
                pred_cat: Optional[Dict[str, Any]] = None) -> bool:
    """
    :param classifier: The RDR classifier to check the prediction of.
    :param case_query: The case query to check.
    :param pred_cat: The predicted category.
    :return: Whether the classifier prediction is matching case_query target or not.
    """
    if case_query.target is None:
        return False
    if pred_cat is None:
        pred_cat = classifier(case_query.case)
    if not isinstance(pred_cat, dict):
        pred_cat = {case_query.attribute_name: pred_cat}
    target = {case_query.attribute_name: case_query.target_value}
    precision, recall = calculate_precision_and_recall(pred_cat, target)
    return all(recall) and all(precision)


def load_or_create_func_rdr_model(func, model_dir: str, rdr_type: Type[RippleDownRules],
                                  **rdr_kwargs) -> RippleDownRules:
    """
    Load the RDR model of the function if it exists, otherwise create a new one.

    :param func: The function to load the model for.
    :param model_dir: The directory where the model is stored.
    :param rdr_type: The type of the RDR model to load.
    :param rdr_kwargs: Additional arguments to pass to the RDR constructor in the case of a new model.
    """
    model_name = get_func_rdr_model_name(func)
    model_path = os.path.join(model_dir, model_name, "rdr_metadata", f"{model_name}.json")
    if os.path.exists(model_path):
        rdr = rdr_type.load(load_dir=model_dir, model_name=model_name)
    else:
        rdr = rdr_type(**rdr_kwargs)
    return rdr


def get_an_updated_case_copy(case: Case, conclusion: Callable, attribute_name: str, conclusion_type: Tuple[Type, ...],
                             mutually_exclusive: bool) -> Case:
    """
    :param case: The case to copy and update.
    :param conclusion: The conclusion to add to the case.
    :param attribute_name: The name of the attribute to update.
    :param conclusion_type: The type of the conclusion to update.
    :param mutually_exclusive: Whether the rule belongs to a mutually exclusive RDR.
    :return: A copy of the case updated with the given conclusion.
    """
    case_cp = copy_case(case)
    temp_case_query = CaseQuery(case_cp, attribute_name, conclusion_type,
                                mutually_exclusive=mutually_exclusive)
    output = conclusion(case_cp)
    if not isinstance(output, Dict):
        output = {attribute_name: output}
    update_case(temp_case_query, output)
    return case_cp
