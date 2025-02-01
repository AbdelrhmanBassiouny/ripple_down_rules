import pandas as pd
from typing_extensions import List

from .ripple_down_rules import Case, Attribute


def create_cases_from_dataframe(df: pd.DataFrame) -> List[Case]:
    """
    Create cases from a pandas dataframe.

    :param df: pandas dataframe
    :return: list of cases
    """
    att_names = df.keys().tolist()
    all_cases = []
    for row in df.iterrows():
        all_att = [Attribute(att, row[1][att]) for att in att_names]
        all_cases.append(Case(all_att))
    return all_cases
