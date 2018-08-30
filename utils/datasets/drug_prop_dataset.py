"""
    File Name:          UnoPytorch/datasets/drug_prop_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/30/18
    Python Version:     3.6.6
    File Description:

"""
import logging
import torch.utils.data as data

DRUG_PROP_FILENAME = 'combined.panther.targets'

logger = logging.getLogger(__name__)


class DrugPropDataset(data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


    """
    Need a set of functions that returns the raw dataframes to avoid re-writing them
    RNASeqDataset, DrugRespDataset, and encoder_init, and here, 
    
    
    """
