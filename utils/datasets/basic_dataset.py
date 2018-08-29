""" 
    File Name:          UnoPytorch/basic_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/16/18
    Python Version:     3.6.6
    File Description:   

"""

import numpy as np
import pandas as pd
import torch.utils.data as data


class DataFrameDataset(data.Dataset):

    def __init__(
            self,
            dataframe: pd.DataFrame,
            ram_dtype: type = np.float16,
            out_dtype: type = np.float32, ):

        self.__data = dataframe.values.astype(ram_dtype)
        self.__out_dtype = out_dtype
        self.__len = len(self.__data)

    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        return self.__data[index].astype(self.__out_dtype)
