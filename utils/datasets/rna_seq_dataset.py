""" 
    File Name:          UnoPytorch/rna_seq_dataset.py
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

from utils.data_processing.dataframe_scaling import scale_dataframe
from utils.miscellaneous.file_downloading import download_files
from utils.data_processing.label_encoding import encode_label_to_int, \
    encode_int_to_onehot, get_labels

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


class RNASeqDataset(data.Dataset):
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
        training (bool): indicator of training/validation dataset
        cells (list): list of all the cells in the dataset
        num_cells (int): number of cell lines in the dataset
        rnaseq_dim (int): dimensionality of RNA sequence
        num_data_src (int): number of data sources overall
    """

    def __init__(
            self,
            data_folder: str,
            training: bool,
            rand_state: int = 0,
            summary: bool = True,

            # Data type settings (for storage and data loading)
            int_dtype: type = np.int8,
            float_dtype: type = np.float16,
            output_dtype: type = np.float32,

            # Pre-processing settings
            rnaseq_scaling: str = 'std',

            # Partitioning (train/validation) and data usage settings
            rnaseq_feature_usage: str = 'combat',
            validation_ratio: float = 0.2, ):
        """dataset = RNASeqDataset('./data/', True)

        Construct a RNA sequence dataset based on the parameters provided.
        The process includes:
            * Downloading source data files;
            * Pre-processing (scaling);
            * Public attributes and other preparations.

        Args:
            data_folder (str): path to data folder.
            training (bool): indicator for training.
            rand_state (int): random seed used for training/validation split
                and other processes that requires randomness.
            summary (bool): set True for printing dataset summary.

            int_dtype (type): integer dtype for data storage in RAM and disk.
            float_dtype (type): float dtype for data storage in RAM and disk.
            output_dtype (type): output dtype for neural network.

            rnaseq_scaling (str): scaling method for RNA squence. Choose
                between 'none', 'std', and 'minmax'.
            rnaseq_feature_usage: RNA sequence data usage. Choose between
                'source_scale' and 'combat'.
            validation_ratio (float): portion of validation data out of all
                data samples.
        """

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

        self.__int_dtype = int_dtype
        self.__int_dtype_str = str(int_dtype).split('\'')[1].split('.')[-1]
        self.__float_dtype = float_dtype
        self.__float_dtype_str = str(float_dtype).split('\'')[1].split('.')[-1]
        self.__output_dtype = output_dtype

        if rnaseq_scaling is None or rnaseq_scaling == '':
            rnaseq_scaling = 'none'
        self.__rnaseq_scaling = rnaseq_scaling.lower()

        self.__rnaseq_feature_usage = rnaseq_feature_usage
        self.__validation_ratio = validation_ratio

        # Download (if necessary) #############################################
        download_files(filenames=FILENAMES,
                       ftp_root=FTP_ROOT,
                       target_folder=self.__raw_data_folder)

        # Processing and load dataframes ######################################
        self.__cl_df = self.__merge_cl_df()
        self.num_data_src = self.__one_hot_encoding()

        # Train/validation split ##############################################
        self.__split_drug_resp()

        # Converting dataframes to arrays for rapid access ####################
        self.__cl_array = self.__cl_df.values.astype(self.__float_dtype)

        # Public attributes ###################################################
        self.cells = self.__cl_df.index.tolist()
        self.num_cells = self.__cl_df.shape[0]
        self.rnaseq_dim = len(self.__cl_df.iloc[0]['seq'])

        # Clear the dataframes ################################################
        self.__cl_df = None

        # Dataset summary #####################################################
        if summary:
            print('=' * 80)
            print(('Training' if self.training else 'Validation')
                  + ' RNA Sequence Dataset Summary:')
            print('\t%i Unique Cell Lines (feature dim: %4i).'
                  % (self.num_cells, self.rnaseq_dim))
            print('=' * 80)

    def __len__(self):
        """length = len(rna_seq_dataset)

        Get the length of dataset, which is the number of cell lines.

        Returns:
            int: the length of dataset.
        """
        return self.num_cells

    def __getitem__(self, index):
        """rnaseq, data_src, site, type, category = rna_seq_dataset[0]

        Args:
            index (int): index for target data slice.

        Returns:
            tuple: a tuple containing the following five elements:
                * RNA sequence data (np.ndarray of float);
                * one-hot-encoded data source (np.ndarray of float);
                * encoded cell line site (int);
                * encoded cell line type (int);
                * encoded cell line category (int)
        """

        cl_data = self.__cl_array[index]

        rnaseq = np.asarray(cl_data[4], dtype=self.__output_dtype)
        data_src = np.array(cl_data[0], dtype=self.__output_dtype)

        # Note that PyTorch requires np.int64 for classification labels
        cl_site = np.int64(cl_data[1])
        cl_type = np.int64(cl_data[2])
        cl_category = np.int64(cl_data[3])

        return rnaseq, data_src, cl_site, cl_type, cl_category

    def __process_meta(self):
        """self.__process_mata()

        This function reads from cell line metadata file and process it into
        dataframe as return.

        Returns:
            pd.dataframe: cell line metadata dataframe.
        """

        logger.info('Processing cell line meta dataframe ... ')

        meta_df_filename = 'cl_meta_df.pkl'
        meta_df_path = os.path.join(self.__processed_data_folder,
                                    meta_df_filename)

        # If the dataframe already exists, read and return
        if os.path.exists(meta_df_path):
            return pd.read_pickle(meta_df_path)

        meta_df = pd.read_csv(
            os.path.join(self.__raw_data_folder, CL_METADATA_FILENAME),
            sep='\t',
            header=0,
            index_col=0,
            usecols=['sample_name',
                     'dataset',
                     'simplified_tumor_site',
                     'simplified_tumor_type',
                     'sample_category'],
            dtype=str)

        # Renaming columns for shorter and better column names
        meta_df.index.names = ['sample']
        meta_df.columns = ['data_src', 'site', 'type', 'category']

        # Transform all the features from text to numeric encoding
        features = meta_df.columns
        dict_filenames = [i + '_dict.json' for i in features]
        for feature, dict_filename in zip(features, dict_filenames):
            meta_df[feature], _ = encode_label_to_int(
                meta_df[feature],
                os.path.join(self.__processed_data_folder, dict_filename))
            meta_df[feature] = meta_df[feature].astype(self.__int_dtype)

        # Save the dataframe into processed folder
        meta_df.to_pickle(meta_df_path)
        return meta_df

    def __process_rnaseq(self):
        """self.__process_rnaseq()

        This function reads from cell line RNA sequence file and process it
        into dataframe as return.

        Returns:
            pd.dataframe: (un-trimmed) RNA sequence dataframe.
        """

        logger.info('Processing RNA sequence dataframe ... ')

        if self.__rnaseq_feature_usage == 'source_scale':
            rnaseq_raw_filename = RNASEQ_SOURCE_SCALE_FILENAME
        elif self.__rnaseq_feature_usage == 'combat':
            rnaseq_raw_filename = RNASEQ_COMBAT_FILENAME
        else:
            logger.error('Unknown RNA feature %s.' %
                         self.__rnaseq_feature_usage, exc_info=True)
            raise ValueError('RNA feature usage must be one of '
                             '\'source_scale\' or \'combat\'.')

        rnaseq_df_filename = 'rnaseq_df(%s, scaling=%s, dtype=%s).pkl' \
                             % (self.__rnaseq_feature_usage,
                                self.__rnaseq_scaling,
                                self.__float_dtype_str)
        rnaseq_df_path = os.path.join(self.__processed_data_folder,
                                      rnaseq_df_filename)

        # If the dataframe already exists, read and return
        if os.path.exists(rnaseq_df_path):
            return pd.read_pickle(rnaseq_df_path)

        # Construct and save the RNA seq dataframe
        rnaseq_df = pd.read_csv(
            os.path.join(self.__raw_data_folder, rnaseq_raw_filename),
            sep='\t',
            header=0,
            index_col=0,
            dtype=str).astype(self.__float_dtype)
        rnaseq_df = scale_dataframe(rnaseq_df, self.__rnaseq_scaling)
        rnaseq_df.to_pickle(rnaseq_df_path)
        return rnaseq_df

    def __merge_cl_df(self):
        """self.__merge_cl_df()

        This function takes RNA sequence dataframe and metadata dataframe
        and merge them into a single dataframe for better accessibility.

        Returns:
            pd.dataframe: RNA sequence dataframe with metadata
        """

        meta_df = self.__process_meta()

        # Put all RNA sequence in a single dataframe cell, preserving the
        # data type and precision
        rnaseq_df = self.__process_rnaseq()
        rnaseq_df['seq'] = list(map(self.__float_dtype,
                                    rnaseq_df.values.tolist()))

        # Join the RNA sequence dataframe with metadata
        cl_df = pd.concat([meta_df, rnaseq_df[['seq']]], axis=1, join='inner')

        return cl_df

    def __one_hot_encoding(self):
        """num_data_src = self.__one_hot_encoding()

        This function takes the integer-encoded data sources from the
        merged cell line dataframe, and convert them into one-hot-encoded
        labels. Note

        Returns:
            int: number of data sources in total.
        """

        num_data_src = len(get_labels(os.path.join(
            self.__processed_data_folder, 'data_src_dict.json')))

        self.__cl_df['data_src'] = encode_int_to_onehot(
            self.__cl_df['data_src'],
            num_classes=num_data_src,
            dtype=self.__int_dtype)

        return num_data_src

    def __split_drug_resp(self):
        """self.__split_drug_resp()

        Split training and validation dataframe for cell lines, stratified
        on tumor type. Note that after the split, our dataframe will only
        contain training/validation data based on training indicator.

        Returns:
            None
        """
        training_cl_df, validation_cl_df = \
            train_test_split(self.__cl_df,
                             test_size=self.__validation_ratio,
                             stratify=self.__cl_df['type'].tolist(),
                             random_state=self.__rand_state)

        self.__cl_df = training_cl_df if self.training else validation_cl_df


# Test segment for cell type dataset
if __name__ == '__main__':

    # Test DrugRespDataset class
    dataloader = torch.utils.data.DataLoader(
        RNASeqDataset(data_folder='../../data/',
                      training=False),
        batch_size=512, shuffle=False)

    tmp = dataloader.dataset[0]
