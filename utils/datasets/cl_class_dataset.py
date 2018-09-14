""" 
    File Name:          UnoPytorch/cl_class_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:   
        This file implements the dataset for cell line classification.
"""

import logging

import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from utils.data_processing.cell_line_dataframes import get_rna_seq_df, \
    get_cl_meta_df
from utils.data_processing.label_encoding import encode_int_to_onehot, \
    get_label_dict

logger = logging.getLogger(__name__)


class CLClassDataset(data.Dataset):
    """Dataset class for cell line classification

    This class implements a PyTorch Dataset class made for cell line
    classification. Using enumerate() or any other methods that utilize
    __getitem__() to access the data.

    Each data item is made of a tuple of
        (RNA_sequence, cell_description, data_source, site, type, category)

    Note that all categorical labels are numeric, and the encoding
    dictionary can be found in the processed folder.

    Attributes:
        training (bool): indicator of training/validation dataset
        cells (list): list of all the cells in the dataset
        num_cells (int): number of cell lines in the dataset
        rnaseq_dim (int): dimensionality of RNA sequence
    """

    def __init__(
            self,
            data_root: str,
            training: bool,
            rand_state: int = 0,
            summary: bool = True,

            # Data type settings (for storage and data loading)
            int_dtype: type = np.int8,
            float_dtype: type = np.float16,
            output_dtype: type = np.float32,

            # Pre-processing settings
            rnaseq_scaling: str = 'std',
            predict_target: str = 'class',

            # Partitioning (train/validation) and data usage settings
            rnaseq_feature_usage: str = 'source_scale',
            validation_ratio: float = 0.2, ):
        """dataset = CLClassDataset('./data/', True)

        Construct a RNA sequence dataset based on the parameters provided.
        The process includes:
            * Downloading source data files;
            * Pre-processing (scaling);
            * Public attributes and other preparations.

        Args:
            data_root (str): path to data root folder.
            training (bool): indicator for training.
            rand_state (int): random seed used for training/validation split
                and other processes that requires randomness.
            summary (bool): set True for printing dataset summary.

            int_dtype (type): integer dtype for data storage in RAM.
            float_dtype (type): float dtype for data storage in RAM.
            output_dtype (type): output dtype for neural network.

            rnaseq_scaling (str): scaling method for RNA sequence. Choose
                between 'none', 'std', and 'minmax'.
            predict_target (str): prediction target for RNA sequence. Note
                that any labels except for target will be in one-hot
                encoding, while the target will be encoded as integers.
                Choose between 'none', 'class', and 'source'.

            rnaseq_feature_usage: RNA sequence data usage. Choose between
                'source_scale' and 'combat'.
            validation_ratio (float): portion of validation data out of all
                data samples.
        """

        # Initialization ######################################################
        self.__data_root = data_root

        # Class-wise variables
        self.training = training
        self.__rand_state = rand_state
        self.__output_dtype = output_dtype

        # Feature scaling
        if rnaseq_scaling is None or rnaseq_scaling == '':
            rnaseq_scaling = 'none'
        self.__rnaseq_scaling = rnaseq_scaling.lower()
        if predict_target is None or predict_target == '':
            predict_target = 'none'
        assert predict_target.lower() in ['none', 'class', 'source']
        self.__predict_target = predict_target.lower()

        self.__rnaseq_feature_usage = rnaseq_feature_usage
        self.__validation_ratio = validation_ratio

        # Load all dataframes #################################################
        self.__rnaseq_df = get_rna_seq_df(
            data_root=data_root,
            rnaseq_feature_usage=rnaseq_feature_usage,
            rnaseq_scaling=rnaseq_scaling,
            float_dtype=float_dtype)

        self.__cl_meta_df = get_cl_meta_df(
            data_root=data_root,
            int_dtype=int_dtype)

        # Put all the sequence in one column as list and specify dtype
        self.__rnaseq_df['seq'] = \
            list(map(float_dtype, self.__rnaseq_df.values.tolist()))

        # Join the RNA sequence data with meta data. cl_df will have columns:
        # ['data_src', 'site', 'type', 'category', 'seq']
        self.__cl_df = pd.concat([self.__cl_meta_df,
                                  self.__rnaseq_df[['seq']]],
                                 axis=1, join='inner')

        # Exclude 'GDC' and 'NCI60' during data source prediction
        # GDC has too many samples while NCI60 has not enough
        if self.__predict_target == 'source':
            logger.warning('Taking out GDC and NCI60 samples to make dataset '
                           'balanced among all data sources ...')
            self.__cl_df = self.__cl_df[
                ~self.__cl_df['data_src'].isin([2, 5])]

        # Encode labels (except for prediction targets) into one-hot encoding
        if self.__predict_target != 'source':
            enc_data_src = encode_int_to_onehot(
                self.__cl_df['data_src'].tolist(),
                len(get_label_dict(data_root, 'data_src_dict.txt')))
            self.__cl_df['data_src'] = list(map(int_dtype, enc_data_src))

        if self.__predict_target != 'class':
            for label in ['site', 'type', 'category']:
                enc_label = encode_int_to_onehot(
                    self.__cl_df[label].tolist(),
                    len(get_label_dict(data_root, '%s_dict.txt' % label)))
                self.__cl_df[label] = list(map(int_dtype, enc_label))

        # Train/validation split ##############################################
        self.__split_drug_resp()

        # Converting dataframes to arrays for rapid access ####################
        self.__cl_array = self.__cl_df.values

        # Public attributes ###################################################
        self.cells = self.__cl_df.index.tolist()
        self.num_cells = self.__cl_df.shape[0]
        self.rnaseq_dim = len(self.__cl_df.iloc[0]['seq'])

        # Clear the dataframes ################################################
        self.__rnaseq_df = None
        self.__cl_meta_df = None
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
        """length = len(cl_class_dataset)

        Get the length of dataset, which is the number of cell lines.

        Returns:
            int: the length of dataset.
        """
        return self.num_cells

    def __getitem__(self, index):
        """rnaseq, data_src, site, type, category = cl_class_dataset[0]

        Note that for all the labels that are not targets, they will be
        encoded as one-hot arrays (np.array(output_dtype)). But if they are
        prediction targets, PyTorch requires them to be encoded and returned
        as integers (np.int64).

        Args:
            index (int): index for target data slice.

        Returns:
            tuple: a tuple containing the following five elements:
                * RNA sequence data (np.array of float);
                * encoded data source (int or np.array of float);
                * encoded cell line site (int or np.array of float);
                * encoded cell line type (int or np.array of float);
                * encoded cell line category (int or np.array of float).
        """

        cl_data = self.__cl_array[index]

        rnaseq = np.asarray(cl_data[4], dtype=self.__output_dtype)

        if self.__predict_target != 'source':
            data_src = np.array(cl_data[0], dtype=self.__output_dtype)
        else:
            data_src = np.int64(cl_data[0])

        if self.__predict_target != 'class':
            cl_site = np.array(cl_data[1], dtype=self.__output_dtype)
            cl_type = np.array(cl_data[2], dtype=self.__output_dtype)
            cl_category = np.array(cl_data[3], dtype=self.__output_dtype)
        else:
            cl_site = np.int64(cl_data[1])
            cl_type = np.int64(cl_data[2])
            cl_category = np.int64(cl_data[3])

        return rnaseq, data_src, cl_site, cl_type, cl_category

    def __split_drug_resp(self):
        """self.__split_drug_resp()

        Split training and validation dataframe for cell lines, stratified
        on tumor type. Note that after the split, our dataframe will only
        contain training/validation data based on training indicator.

        Returns:
            None
        """
        split_kwargs = {
            'test_size': self.__validation_ratio,
            'random_state': self.__rand_state,
            'shuffle': True, }

        try:
            training_cl_df, validation_cl_df = \
                train_test_split(self.__cl_df, **split_kwargs,
                                 stratify=self.__cl_df['type'].tolist())
        except ValueError:
            logger.warning('Failed to split cell lines in stratified way. '
                           'Splitting randomly ...')
            training_cl_df, validation_cl_df = \
                train_test_split(self.__cl_df, **split_kwargs)

        self.__cl_df = training_cl_df if self.training else validation_cl_df


# Test segment for cell line classification dataset
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    # Test DrugRespDataset class
    dataloader = torch.utils.data.DataLoader(
        CLClassDataset(data_root='../../data/',
                       predict_target='source',
                       training=False),
        batch_size=512, shuffle=False)

    # print(dataloader.dataset[0])
