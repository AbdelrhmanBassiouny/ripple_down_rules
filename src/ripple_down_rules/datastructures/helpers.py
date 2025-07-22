from __future__ import annotations

from typing import Any

from .case import Case


def get_case_name(case: Any) -> str:
    """
    Get the case name from the case object or query.

    :param case: The case object or query.
    :return: The name of the case.
    """
    return case._name if isinstance(case, Case) else case.__class__.__name__
