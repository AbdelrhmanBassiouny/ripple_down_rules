import os

from sqlalchemy.orm import Session
from typing_extensions import Type, Optional

from ripple_down_rules.rdr import RippleDownRules
from ripple_down_rules.utils import get_func_rdr_model_path


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
