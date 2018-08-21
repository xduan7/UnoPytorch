""" 
    File Name:          UnoPytorch/cell_line_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:   

"""

import os
import errno
import logging

import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from utils.miscellaneous.file_downloading import download_files
from utils.miscellaneous.dataframe_scaling import scale_dataframe
from utils.miscellaneous.label_encoding import encode_labels, \
    get_label_encoding_dict, get_labels


# FTP address and filenames
FTP_ROOT = 'http://ftp.mcs.anl.gov/pub/candle/public/' \
           'benchmarks/Pilot1/combo/'

CL_METADATA_FILENAME = 'combined_cl_metadata'
RNASEQ_COMBAT_FILENAME = 'combined_rnaseq_data_lincs1000_combat'
RNASEQ_SOURCE_SCALE_FILENAME = 'combined_rnaseq_data_lincs1000_source_scale'

FILENAMES = [
    CL_METADATA_FILENAME,
    RNASEQ_COMBAT_FILENAME,
    RNASEQ_SOURCE_SCALE_FILENAME
]

# Local folders in data root
RAW_FOLDER = './raw/'
PROCESSED_FOLDER = './processed/'

logger = logging.getLogger(__name__)


class CellLineDataset(data.Dataset):
    """Dataset class for cell line classification

    This class implements a PyTorch Dataset class made for cell line
    classification. Using enumerate() or any other methods that utilize
    __getitem__() to access the data.

    Each data item is made of a tuple of
        (RNA_sequence, conditions, site, type, category)
    where conditions is a list of [data_source, cell_description].

    Note that all categorical labels are numeric, and the encoding
    dictionary can be found in the processed folder.

    Attributes:
        rnaseq_dim (int): dimensionality of RNA sequence.
    """

    def __init__(
            self,
            data_folder: str,
            training: bool,
            rand_state: int = 0,

            # Data type settings (for storage and data loading)
            float_dtype: type = np.float16,

            # Pre-processing settings
            rnaseq_scaling: str = 'none',

            # Partitioning (train/validation) and data usage settings
            rnaseq_feature_usage: str = 'source_scale',
            validation_size: float = 0.2):

        # Initialization ######################################################
        self.__data_folder = data_folder
        self.__raw_data_folder = \
            os.path.join(self.__data_folder, RAW_FOLDER)
        self.__processed_data_folder = \
            os.path.join(self.__data_folder, PROCESSED_FOLDER)

        logger.info('Creating raw & processed data folder ... ')
        for folder in [self.__raw_data_folder, self.__processed_data_folder]:
            try:
                os.makedirs(folder)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    logger.error('Failed to create data folders',
                                 exc_info=True)
                    raise

        # Class-wise variables
        self.training = training
        self.__rand_state = rand_state

        self.__float_dtype = float_dtype
        self.__float_dtype_str = str(float_dtype).split('\'')[1].split('.')[-1]

        if rnaseq_scaling is None or rnaseq_scaling == '':
            rnaseq_scaling = 'none'
        self.__rnaseq_scaling = rnaseq_scaling.lower()

        self.__rnaseq_feature_usage = rnaseq_feature_usage
        self.__validation_size = validation_size

        # Download (if necessary) #############################################
        download_files(filenames=FILENAMES,
                       ftp_root=FTP_ROOT,
                       target_folder=self.__raw_data_folder)

        # Processing and load dataframes ######################################
        self.__cl_df = self.__process_cell_line()


    def __getitem__(self, index):
        """rnaseq, conditions, site, type, category = \
                cell_type_dataset.__getitem__(index=0)
        This function fetches the feature, condition and target with the
        corresponding index of cell line dataset.
        Args:
            index (int): index for cell line dataset.
        Returns:
            tuple: RNA sequence (list of float), conditions (list of float),
                site (float), type (float), category (float).
        """

        cl_data = self._df.iloc[index]

        rnaseq = np.array(cl_data[self._rna_list], dtype=np.float32)
        data_src = np.array([cl_data['data_src'], ], dtype=np.float32)

        # conditions = np.array(cl_data[['data_src', 'description']],
        #                       dtype=np.float32)

        #cl_site = np.array([cl_data['site'], ], dtype=np.float32)
        # cl_type = np.array([cl_data['type'], ], dtype=np.float32)
        # cl_category = np.array([cl_data['category'], ], dtype=np.float32)

        cl_site = np.int64(cl_data['site'])
        cl_type = np.int64(cl_data['type'])
        cl_category = np.int64(cl_data['category'])

        return rnaseq, data_src, cl_site, cl_type, cl_category

    def __len__(self):
        """length = cell_type_dataset.__len__()

        Get the length of dataset, which is the number of cell lines.

        Returns:
            int: the length of dataset.
        """
        return self._df.shape[0]


# Test segment for cell type dataset
if __name__ == '__main__':

    data_sources = get_labels('../../data/shared/data_src_dict.json')
    sites = get_labels('../../data/shared/site_dict.json')
    types = get_labels('../../data/shared/type_dict.json')
    categories = get_labels('../../data/shared/category_dict.json')

    print('Data Sources (%d): %r' % (len(data_sources), data_sources))
    print('Cell Line Sites (%d): %r' % (len(sites), sites))
    print('Cell Line Types (%d): %r' % (len(types), types))
    print('Cell Line Categories (%d): %r' % (len(categories), categories))

    # Test DrugRespDataset class
    dataloader = torch.utils.data.DataLoader(
        CellTypeDataset(data_root='../../data/',
                        data_sources='all',
                        data_portion='testing',
                        verbose=False),
        batch_size=512, shuffle=False)

    data, cond, target_site, target_type, target_category \
        = dataloader.dataset[0]

    print('Dataset Loaded.\n'
          '\tRNA Sequence Dimension:\t %6d;\n'
          '\tNumber of Samples:\t %10d;'
          % (len(data), len(dataloader.dataset)))

    # Use next(iter(dataloader)) to get the next batch.
    # Note that shuffle must be True, otherwise every batch is the same