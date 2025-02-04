import pandas as pd
from typing_extensions import List

from .rdr import Case, Attribute


def create_cases_from_dataframe(df: pd.DataFrame, ids: List[str]) -> List[Case]:
    """
    Create cases from a pandas dataframe.

    :param df: pandas dataframe
    :param ids: list of ids
    :return: list of cases
    """
    att_names = df.keys().tolist()
    all_cases = []
    for _id, row in zip(ids, df.iterrows()):
        all_att = [Attribute(att, row[1][att]) for att in att_names]
        all_cases.append(Case(_id, all_att))
    return all_cases
