import pandas as pd
from typing_extensions import List, Dict, Type

from .datastructures import Case, Attribute, Unary, Bool, Categorical, Integer, Continuous


def create_cases_from_dataframe(df: pd.DataFrame, ids: List[str]) -> List[Case]:
    """
    Create cases from a pandas dataframe.

    :param df: pandas dataframe
    :param ids: list of ids
    :return: list of cases
    """
    att_names = df.keys().tolist()
    unique_values: Dict[str, List] = {col_name: df[col_name].unique() for col_name in att_names}
    att_types: Dict[str, Type[Attribute]] = {}
    for col_name, values in unique_values.items():
        values = values.tolist()
        if len(values) == 1:
            att_types[col_name] = type(col_name, (Unary,), {})
        elif len(values) == 2 and all(isinstance(val, bool) or (val in [0, 1]) for val in values):
            att_types[col_name] = type(col_name, (Bool,), {})
        elif len(values) >= 2 and all(isinstance(val, str) for val in values):
            att_types[col_name] = type(col_name, (Categorical,), {'_range': set(values)})
            att_types[col_name].create_values()
        elif len(values) >= 2 and all(isinstance(val, int) for val in values):
            att_types[col_name] = type(col_name, (Integer,), {})
        elif len(values) >= 2 and all(isinstance(val, float) for val in values):
            att_types[col_name] = type(col_name, (Continuous,), {})
    all_cases = []
    for _id, row in zip(ids, df.iterrows()):
        all_att = [att_types[att](row[1][att]) for att in att_names]
        all_cases.append(Case(_id, all_att))
    return all_cases
