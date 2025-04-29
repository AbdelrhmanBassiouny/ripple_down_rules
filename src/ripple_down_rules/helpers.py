from __future__ import annotations

import os

from .datastructures.dataclasses import CaseQuery
from sqlalchemy.orm import Session
from typing_extensions import Type, Optional, Callable, Any, Dict, TYPE_CHECKING

from .utils import get_func_rdr_model_path
from .utils import calculate_precision_and_recall

if TYPE_CHECKING:
    from .rdr import RippleDownRules


def is_matching(classifier: Callable[[Any], Any], case_query: CaseQuery, pred_cat: Optional[Dict[str, Any]] = None) -> bool:
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
                                  session: Optional[Session] = None, **rdr_kwargs) -> RippleDownRules:
    """
    Load the RDR model of the function if it exists, otherwise create a new one.

    :param func: The function to load the model for.
    :param model_dir: The directory where the model is stored.
    :param rdr_type: The type of the RDR model to load.
    :param session: The SQLAlchemy session to use.
    :param rdr_kwargs: Additional arguments to pass to the RDR constructor in the case of a new model.
    """
    model_path = get_func_rdr_model_path(func, model_dir)
    if os.path.exists(model_path):
        rdr = rdr_type.load(model_path)
        rdr.session = session
    else:
        rdr = rdr_type(session=session, **rdr_kwargs)
    return rdr
