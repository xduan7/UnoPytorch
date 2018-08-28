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
from utils.data_processing.dataframe_to_dict import df_to_dict


class DataFrameDataset(data.Dataset):

    def __init__(

            self,
            dataframe: pd.DataFrame,
            dtype: type = np.float32, ):

        self.__dataframe = dataframe
        self.__len = len(dataframe)
        self.__dtype = dtype

        self.__dict = df_to_dict(self.__dataframe, self.__dtype)

    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        return self.__dict[index]
