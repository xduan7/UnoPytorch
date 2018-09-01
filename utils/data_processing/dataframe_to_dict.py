""" 
    File Name:          UnoPytorch/dataframe_to_dict.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/23/18
    Python Version:     3.6.6
    File Description:   

"""
import sys

import pandas as pd


def df_to_dict(df: pd.DataFrame, dtype: type):

    return {idx: row.values.astype(dtype) for idx, row in df.iterrows()}
