""" 
    File Name:          UnoPytorch/drug_resp_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:   

"""
import os
import errno
import logging
import multiprocessing
import time

from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from utils.miscellaneous.dataframe_to_dict import df_to_dict
from utils.miscellaneous.file_downloading import download_files
from utils.miscellaneous.dataframe_scaling import scale_dataframe
from utils.miscellaneous.label_encoding import encode_label_to_int, \
    get_label_encoding_dict


# FTP address and filenames
FTP_ROOT = 'http://ftp.mcs.anl.gov/pub/candle/public/' \
           'benchmarks/Pilot1/combo/'

DRUG_RESP_FILENAME = 'rescaled_combined_single_drug_growth'
ECFP_FILENAME = 'pan_drugs_dragon7_ECFP.tsv'
PFP_FILENAME = 'pan_drugs_dragon7_PFP.tsv'
DESCRIPTOR_FILENAME = 'pan_drugs_dragon7_descriptors.tsv'
DRUG_INFO_FILENAME = 'drug_info'
CL_METADATA = 'combined_cl_metadata'
RNASEQ_SOURCE_SCALE_FILENAME = 'combined_rnaseq_data_lincs1000_source_scale'
RNASEQ_COMBAT_FILENAME = 'combined_rnaseq_data_lincs1000_combat'

# List of all filenames
FILENAMES = [
    DRUG_RESP_FILENAME,

    DRUG_INFO_FILENAME,
    ECFP_FILENAME,
    PFP_FILENAME,
    DESCRIPTOR_FILENAME,

    CL_METADATA,
    RNASEQ_SOURCE_SCALE_FILENAME,
    RNASEQ_COMBAT_FILENAME, ]

# Local folders in data root
RAW_FOLDER = './raw/'
PROCESSED_FOLDER = './processed/'

logger = logging.getLogger(__name__)


class DrugRespDataset(data.Dataset):
    """Dataset class for drug response learning

    This class implements a PyTorch Dataset class made for drug response
    learning. Using enumerate() or any other methods that utilize
    __getitem__() to access the data.

    Each data item is made of a tuple of (feature, target), where feature is
    a list including drug and cell line information along with the log
    concentration, and target is the growth.

    Note that all items in feature and the target are in python float type.

    Attributes:
        training (bool): indicator of training/validation dataset
        drugs (list): list of all the drugs in the dataset
        cells (list): list of all the cells in the dataset
        data_source (str): source of the data being used.
        num_records (int): number of drug response records.
        drug_feature_dim (int): dimensionality of drug feature.
        rnaseq_dim (int): dimensionality of RNA sequence.
    """

    def __init__(

            self,
            data_folder: str,
            data_source: str,
            training: bool,
            rand_state: int = 0,
            summary: bool = True,

            # Data type settings (for storage and data loading)
            int_dtype: type = np.int8,
            float_dtype: type = np.float16,
            output_dtype: type = np.float32,

            # Pre-processing settings
            growth_scaling: str = 'none',
            descriptor_scaling: str = 'std',
            rnaseq_scaling: str = 'std',
            nan_threshold: float = 0.0,

            # Partitioning (train/validation) and data usage settings
            rnaseq_feature_usage: str = 'combat',
            drug_feature_usage: str = 'both',
            validation_size: float = 0.2,
            disjoint_drugs: bool = True,
            disjoint_cells: bool = True, ):

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
        self.data_source = data_source
        self.training = training
        self.__rand_state = rand_state

        self.__int_dtype = int_dtype
        self.__int_dtype_str = str(int_dtype).split('\'')[1].split('.')[-1]
        self.__float_dtype = float_dtype
        self.__float_dtype_str = str(float_dtype).split('\'')[1].split('.')[-1]
        self.__output_dtype = output_dtype

        # Feature scaling
        if growth_scaling is None or growth_scaling == '':
            growth_scaling = 'none'
        self.__growth_scaling = growth_scaling.lower()
        if descriptor_scaling is None or descriptor_scaling == '':
            self.__descriptor_scaling = 'none'
        self.__descriptor_scaling = descriptor_scaling
        if rnaseq_scaling is None or rnaseq_scaling == '':
            self.__rnaseq_scaling = 'none'
        self.__rnaseq_scaling = rnaseq_scaling

        self.__nan_threshold = nan_threshold

        self.__rnaseq_feature_usage = rnaseq_feature_usage
        self.__drug_feature_usage = drug_feature_usage
        self.__validation_size = validation_size
        self.__disjoint_drugs = disjoint_drugs
        self.__disjoint_cells = disjoint_cells

        # Download (if necessary) #############################################
        download_files(filenames=FILENAMES,
                       ftp_root=FTP_ROOT,
                       target_folder=self.__raw_data_folder)

        # Processing and load dataframes ######################################
        self.__drug_resp_df = self.__process_drug_resp()
        self.__drug_feature_df = self.__process_drug_feature()
        self.__rnaseq_df = self.__process_rnaseq()

        # Trim the dataframes so that they share the same drugs/cells
        self.__trim_dataframes(trim_data_source=False)

        # Train/validation split ##############################################
        self.__split_drug_resp()

        # Converting dataframes to arrays and dict for rapid access ###########
        self.__drug_resp_array = self.__drug_resp_df.values
        self.__drug_feature_dict = df_to_dict(self.__drug_feature_df,
                                              dtype=self.__float_dtype)
        self.__rnaseq_dict = df_to_dict(self.__rnaseq_df,
                                        dtype=self.__float_dtype)

        # Public attributes ###################################################
        self.drugs = self.__drug_resp_df['DRUG_ID'].unique().tolist()
        self.cells = self.__drug_resp_df['CELLNAME'].unique().tolist()
        self.num_records = len(self.__drug_resp_df)
        self.drug_feature_dim = self.__drug_feature_df.shape[1]
        self.rnaseq_dim = self.__rnaseq_df.shape[1]

        # Clear the dataframes ################################################
        self.__drug_resp_df = None
        self.__drug_feature_df = None
        self.__rnaseq_df = None

        # Dataset summary #####################################################
        if summary:
            print('=' * 80)
            print(('Training' if self.training else 'Validation')
                  + ' Drug Response Dataset Summary (Data Source: %6s):'
                  % self.data_source)
            print('\t%i Drug Response Records .' % len(self.__drug_resp_array))
            print('\t%i Unique Drugs (feature dim: %4i).'
                  % (len(self.drugs), self.drug_feature_dim))
            print('\t%i Unique Cell Lines (feature dim: %4i).'
                  % (len(self.cells), self.rnaseq_dim))
            print('=' * 80)

    def __len__(self):
        return self.num_records

    def __getitem__(self, index):

        drug_resp = self.__drug_resp_array[index]

        drug_feature = np.array(self.__drug_feature_dict[drug_resp[1]],
                                dtype=self.__output_dtype)
        rnaseq = np.array(self.__rnaseq_dict[drug_resp[2]],
                          dtype=self.__output_dtype)

        concentration = np.array([drug_resp[3]], dtype=self.__output_dtype)
        growth = np.array([drug_resp[4]], dtype=self.__output_dtype)

        return rnaseq, drug_feature, concentration, growth

    def __process_drug_resp(self):
        """self.__process_drug_resp()

        This function reads from the raw drug response file and process it
        into dataframe as return.
        During the processing, the data sources will be converted to numeric
        and the the growth might be scaled accordingly.

        :return:
            (pd.dataframe): (un-trimmed) drug response dataframe.
        """

        logger.info('Processing drug response dataframe ... ')

        drug_resp_df_filename = \
            'drug_resp_df(growth_scaling=%s, dtype=%s).pkl' \
            % (self.__growth_scaling, self.__float_dtype_str)
        drug_resp_df_path = \
            os.path.join(self.__processed_data_folder, drug_resp_df_filename)

        # If the dataframe already exists, read and return
        if os.path.exists(drug_resp_df_path):
            return pd.read_pickle(drug_resp_df_path)

        # Otherwise load from raw files and process correspondingly
        drug_resp_df = pd.read_csv(
            os.path.join(self.__raw_data_folder, DRUG_RESP_FILENAME),
            sep='\t',
            header=0,
            index_col=None,
            usecols=[0, 1, 2, 4, 6, ],
            dtype=str)

        # Encode data sources into numeric
        data_src_dict_path = os.path.join(
            self.__processed_data_folder, 'data_src_dict.json')
        drug_resp_df['SOURCE'] = encode_label_to_int(
            drug_resp_df['SOURCE'], data_src_dict_path)

        # Scaling the growth
        drug_resp_df['GROWTH'] = scale_dataframe(
            drug_resp_df['GROWTH'], self.__growth_scaling)

        # Convert data source, concentration and growth to numeric
        drug_resp_df[drug_resp_df.columns[0]] = \
            drug_resp_df[drug_resp_df.columns[0]].astype(self.__float_dtype)
        drug_resp_df[drug_resp_df.columns[3:]] = \
            drug_resp_df[drug_resp_df.columns[3:]].astype(self.__float_dtype)

        # Save the dataframe into processed folder
        drug_resp_df.to_pickle(drug_resp_df_path)
        return drug_resp_df

    def __process_drug_feature(self):
        """self.__process_drug_feature()

        This function reads from the raw drug feature files (fingerprint
        and/or descriptor) and process it into dataframe as return.

        :return:
            (pd.dataframe): (un-trimmed) drug feature dataframe.
        """

        logger.info('Processing drug feature dataframe(s) ... ')

        if self.__drug_feature_usage == 'both':
            return pd.concat([self.__process_drug_fingerprint(),
                              self.__process_drug_descriptor()],
                             axis=1, join='inner')
        elif self.__drug_feature_usage == 'fingerprint':
            return self.__process_drug_fingerprint()
        elif self.__drug_feature_usage == 'descriptor':
            return self.__process_drug_descriptor()
        else:
            logger.error('Drug feature must be \'fingerprint\', \'descriptor\''
                         ', or \'both\',', exc_info=True)
            raise ValueError('Undefined drug feature %s.'
                             % self.__drug_feature_usage)

    def __process_drug_fingerprint(self):
        """self.__process_drug_fingerprint()

        This function reads from the raw drug fingerprint file and process it
        into dataframe as return.

        :return:
            (pd.dataframe): (un-trimmed) drug fingerprint dataframe.
        """
        """self._process_drug_fingerprint()

        Process drug fingerprint data file and save the dataframe.

        Returns:
            (pd.dataframe): combined fingerprint dataframe.
        """

        logger.info('Processing drug fingerprint dataframe ... ')

        drug_fingerprint_df_filename = \
            'drug_fingerprint_df(dtype=%s).pkl' % self.__int_dtype_str
        drug_fingerprint_df_path = os.path.join(self.__processed_data_folder,
                                                drug_fingerprint_df_filename)

        # If the dataframe already exists, read and return
        if os.path.exists(drug_fingerprint_df_path):
            return pd.read_pickle(drug_fingerprint_df_path)

        ecfp_df = pd.read_csv(
            os.path.join(self.__raw_data_folder, ECFP_FILENAME),
            sep='\t',
            header=None,
            index_col=0,
            dtype=str,
            skiprows=[0, ]).astype(self.__int_dtype)

        pfp_df = pd.read_csv(
            os.path.join(self.__raw_data_folder, PFP_FILENAME),
            sep='\t',
            header=None,
            index_col=0,
            dtype=str,
            skiprows=[0, ]).astype(self.__int_dtype)

        drug_fingerprint_df = \
            pd.concat([ecfp_df, pfp_df], axis=1, join='inner')

        # Save the dataframe into processed folder
        drug_fingerprint_df.to_pickle(drug_fingerprint_df_path)

        return drug_fingerprint_df

    def __process_drug_descriptor(self):
        """self.__process_drug_descriptor()

        This function reads from the raw drug descriptor file and process it
        into dataframe as return.
        During the processing, columns (features) and rows (drugs) with
        exceeding percentage of NaN values will be dropped. And the feature
        will be scaled accordingly.

        :return:
            (pd.dataframe): (un-trimmed) drug descriptor dataframe.
        """

        logger.info('Processing drug descriptor dataframe ... ')

        drug_descriptor_df_filename = \
            'drug_descriptor_df(scaling=%s, nan_thresh=%.2f, dtype=%s).pkl' \
            % (self.__descriptor_scaling,
               self.__nan_threshold,
               self.__float_dtype_str)
        drug_descriptor_df_path = os.path.join(self.__processed_data_folder,
                                               drug_descriptor_df_filename)

        # If the dataframe already exists, read and return
        if os.path.exists(drug_descriptor_df_path):
            return pd.read_pickle(drug_descriptor_df_path)

        # Note that np.float64 is used to prevent overflow
        drug_descriptor_df = pd.read_csv(
            os.path.join(self.__raw_data_folder, DESCRIPTOR_FILENAME),
            sep='\t',
            header=0,
            index_col=0,
            na_values='na',
            dtype=str).astype(np.float64)

        # Drop NaN values if the percentage of NaN exceeds nan_threshold
        logger.debug('Dropping NaN values for descriptors ...')

        valid_thresh = 1 - self.__nan_threshold

        drug_descriptor_df.dropna(
            axis=1, inplace=True,
            thresh=int(drug_descriptor_df.shape[0] * valid_thresh))

        drug_descriptor_df.dropna(
            axis=0, inplace=True,
            thresh=int(drug_descriptor_df.shape[1] * valid_thresh))

        # Fill the rest of NaN with column means
        drug_descriptor_df.fillna(drug_descriptor_df.mean(), inplace=True)

        # Scaling the descriptor
        logger.debug('Scaling drug descriptors ...')
        drug_descriptor_df = scale_dataframe(
            drug_descriptor_df, self.__descriptor_scaling)

        # Convert to the proper data type and save
        drug_descriptor_df \
            = drug_descriptor_df.astype(self.__float_dtype)
        drug_descriptor_df.to_pickle(drug_descriptor_df_path)

    def __process_rnaseq(self):
        """self.__process_rnaseq()

        This function reads from cell line RNA sequence file and process it
        into dataframe as return.

        :return:
            (pd.dataframe): (un-trimmed) RNA sequence dataframe.
        """

        logger.info('Processing RNA sequence dataframe ... ')

        if self.__rnaseq_feature_usage == 'source_scale':
            rnaseq_raw_filename = RNASEQ_SOURCE_SCALE_FILENAME
        elif self.__rnaseq_feature_usage == 'combat':
            rnaseq_raw_filename = RNASEQ_COMBAT_FILENAME
        else:
            logger.error('Unknown RNA feature %s.'
                         % self.__rnaseq_feature_usage, exc_info=True)
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

    def __trim_dataframes(self, trim_data_source: bool=True):
        """self.__trim_dataframes(trim_data_source=True)

        This function trims three dataframes to make sure that they share
        the same list of drugs and cell lines, in order to avoid wasting
        memory and processing time.
        If trim_data_source is set to True, the function will only take the
        drug response records from such data source, which should be given
        during initialization.


        :param trim_data_source (bool): indicator for data source limitation
        """

        logger.info('Trimming dataframes ... ')

        # Encode the data source and take the data from target source only
        # Note that source could be 'NCI60', 'GDSC', etc and 'all'
        if (self.data_source.lower() != 'all') and trim_data_source:

            logger.debug(
                'Using drug responses from %s ... ' % self.data_source)

            data_src_dict = get_label_encoding_dict(os.path.join(
                self.__processed_data_folder, 'data_src_dict.json'), None)
            try:
                encoded_data_source = data_src_dict[self.data_source]
                self.__drug_resp_df = self.__drug_resp_df.loc[
                    self.__drug_resp_df['SOURCE'] == encoded_data_source]
            except KeyError:
                logger.error('Data source %s not found.' % self.data_source,
                             exc_info=True)
                raise

        # Make sure that all three dataframes share the same drugs/cells
        logger.debug('Trimming on the cell lines and drugs ... ')

        cell_set = set(self.__drug_resp_df['CELLNAME'].unique()) \
                   & set(self.__rnaseq_df.index.values)
        drug_set = set(self.__drug_resp_df['DRUG_ID'].unique()) \
                   & set(self.__drug_feature_df.index.values)

        self.__drug_resp_df = self.__drug_resp_df.loc[
            (self.__drug_resp_df['CELLNAME'].isin(cell_set)) &
            (self.__drug_resp_df['DRUG_ID'].isin(drug_set))]

        self.__drug_resp_df = self.__drug_resp_df.reset_index(drop=True)

        self.__rnaseq_df = self.__rnaseq_df[
            self.__rnaseq_df.index.isin(cell_set)]
        self.__drug_feature_df = self.__drug_feature_df[
            self.__drug_feature_df.index.isin(drug_set)]

        logger.info('There are %i drugs and %i cell lines, with %i response '
                    'records.' % (len(drug_set), len(cell_set),
                                  len(self.__drug_resp_df)))
        return

    def __analyze_drugs(self):
        """
        TODO



        :return:
        """

        logger.info('Analyzing drug + cell combos statistics ... ')

        combo_df_filename = 'drug_cell_combo_stats_df' \
                            '(growth_scaling=%s, dtype=%s).pkl' \
                            % (self.__growth_scaling, self.__float_dtype_str)
        combo_df_path = os.path.join(self.__processed_data_folder,
                                     combo_df_filename)

        columns = ['COMBO', 'DRUG_ID', 'CELLNAME',
                   'NUM_REC', 'AVG', 'VAR', 'CORR', ]

        if os.path.exists(combo_df_path):
            logger.debug('Loading existing drug + cell combo dataframe ... ')
            combo_df = pd.read_pickle(combo_df_path)
        else:
            logger.debug('Constructing drug + cell combo dataframe ... ')
            combo_df = pd.DataFrame(columns=columns)

        # Get all the combos from drug reps dataframe
        tmp_drug_resp_df = self.__drug_resp_df.copy(deep=True)
        tmp_drug_resp_df['COMBO'] = \
            tmp_drug_resp_df['DRUG_ID'] + '+' + tmp_drug_resp_df['CELLNAME']
        tmp_drug_resp_df.set_index('COMBO', inplace=True)
        total_combos = set(tmp_drug_resp_df.index.unique())

        # If there are existing combos not in the current dataframe
        # Iterate through and fill them in
        curr_combos = set(combo_df['COMBO'].tolist())
        # assert curr_combos.issubset(total_combos)
        logger.debug('There are %i (out of %i) combos to process.'
                     % (len(total_combos - curr_combos),
                        len(total_combos)))

        def process_combo(combo: str, single_combo_df: pd.DataFrame):

            drug, cell = combo.split('+')

            # Get all the drug response data from a single drug + cell combo
            # single_combo_df = tmp_drug_resp_df.loc[combo]
            growth_series = single_combo_df['GROWTH']

            num_rec = len(single_combo_df)
            avg = growth_series.mean()
            var = growth_series.var()
            corr = growth_series.corr(single_combo_df['LOG_CONCENTRATION'])

            return [combo, drug, cell, num_rec, avg, var, corr]

        # Parallelize the drug + cell combination processing
        num_cores = multiprocessing.cpu_count()
        combo_list = Parallel(n_jobs=num_cores)(
            delayed(process_combo)(combo, tmp_drug_resp_df.loc[combo])
            for combo in (total_combos - curr_combos))

        # Append the constructed table for combos
        combo_df = combo_df.append(pd.DataFrame(combo_list, columns=columns),
                                   ignore_index=True)

        if len(total_combos - curr_combos) != 0:

            # Set the correct data types and save
            logger.debug('Re-writing the combo dataframe ... ')

            combo_df[['COMBO', 'DRUG_ID', 'CELLNAME']] = \
                combo_df[['COMBO', 'DRUG_ID', 'CELLNAME']].astype(str)
            combo_df[['NUM_REC']] = combo_df[['NUM_REC']].astype(int)
            combo_df[['AVG', 'VAR', 'CORR']] = \
                combo_df[['AVG', 'VAR', 'CORR']].astype(self.__float_dtype)

            if os.path.exists(combo_df_path):
                os.remove(combo_df_path)
            combo_df.to_pickle(combo_df_path)

        logger.info('Analyzing drug statistics ... ')

        total_drugs = set(combo_df['DRUG_ID'].unique())

        drug_df_filename = 'drug_stats_df(growth_scaling=%s, dtype=%s).pkl' \
                           % (self.__growth_scaling, self.__float_dtype_str)
        drug_df_path = os.path.join(self.__processed_data_folder,
                                    drug_df_filename)

        if os.path.exists(drug_df_path):
            logger.debug('Loading existing drug statistics dataframe ... ')
            drug_df = pd.read_pickle(drug_df_path)
        else:
            logger.debug('Constructing drug statistics dataframe ... ')
            drug_df = pd.DataFrame(
                columns=['DRUG_ID', 'NUM_CL', 'NUM_REC', 'AVG', 'CORR', ])

        # If there are existing combos not in the current dataframe
        # Iterate through and fill them in
        curr_drugs = set(drug_df['DRUG_ID'].tolist())
        assert curr_drugs.issubset(total_drugs)
        logger.debug('There are %i (out of %i) drugs to process.'
                     % (len(total_drugs - curr_drugs),
                        len(total_drugs)))

        def process_drug(drug: str, single_drug_df: pd.DataFrame):

            # Get all the drug response data from a single drug + cell combo
            # single_combo_df = tmp_drug_resp_df.loc[combo]
            num_cl = len(single_drug_df)
            rec = single_drug_df['NUM_REC'].values
            avg = single_drug_df['AVG'].values
            corr = np.nan_to_num(single_drug_df['CORR'].values)
            avg_rec = np.mean(rec)

            return [drug, num_cl, avg_rec,
                    np.mean(np.multiply(rec, avg) / avg_rec),
                    np.mean(np.multiply(rec, corr) / avg_rec)]

        # Parallelize the drug stats processing
        tmp_combo_df = combo_df.set_index('DRUG_ID', inplace=False)
        num_cores = multiprocessing.cpu_count()
        drug_list = Parallel(n_jobs=num_cores)(
            delayed(process_drug)(drug, tmp_combo_df.loc[drug])
            for drug in (total_drugs - curr_drugs))

        # Append the constructed table for combos
        drug_df = drug_df.append(
            pd.DataFrame(drug_list,
                         columns=['DRUG_ID', 'NUM_CL', 'NUM_REC',
                                  'AVG', 'CORR', ]),
            ignore_index=True)

        if len(total_drugs - curr_drugs) != 0:

            # Set the correct data types and save
            logger.debug('Re-writing the drug dataframe ... ')

            drug_df[['DRUG_ID']] = drug_df[['DRUG_ID']].astype(str)
            drug_df[['NUM_CL']] = drug_df[['NUM_CL']].astype(int)
            drug_df[['NUM_REC', 'AVG', 'CORR']] = \
                drug_df[['NUM_REC', 'AVG', 'CORR']].astype(self.__float_dtype)

            if os.path.exists(drug_df_path):
                os.remove(drug_df_path)
            drug_df.to_pickle(drug_df_path)

        logger.info('Classifying drugs ... ')

        drugs = drug_df['DRUG_ID'].values
        avg = drug_df['AVG'].values
        corr = drug_df['CORR'].values

        drug_growth = (avg > np.mean(avg))
        drug_corr = (corr > np.mean(corr))

        drug_analysis_array = \
            np.array([drugs, drug_growth, drug_corr]).transpose()

        drug_analysis_df = pd.DataFrame(
            drug_analysis_array,
            columns=['DRUG_ID', 'HIGH_GROWTH', 'HIGH_CORR'])

        drug_analysis_df.set_index('DRUG_ID', inplace=True)

        return drug_analysis_df

    def __load_cl_metadata(self):

        cl_meta_df = pd.read_csv(
            os.path.join(self.__raw_data_folder, CL_METADATA),
            sep='\t',
            header=0,
            index_col=0,
            usecols=['sample_name',
                     'dataset',
                     'simplified_tumor_site',
                     'simplified_tumor_type',
                     'sample_category',
                     'sample_descr'],
            dtype=str)

        # Renaming columns for shorter and better column names
        cl_meta_df.index.names = ['sample']
        cl_meta_df.columns = \
            ['data_src', 'site', 'type', 'category', 'description', ]

        return cl_meta_df

    def __split_drug_resp(self):

        drug_analysis_df = self.__analyze_drugs()
        # cl_meta_df = self.__load_cl_metadata()

        # Trim dataframes based on data source
        self.__trim_dataframes(trim_data_source=True)

        # Get lists of all drugs & cells corresponding from data source
        cell_list = self.__drug_resp_df['CELLNAME'].unique().tolist()
        drug_list = self.__drug_resp_df['DRUG_ID'].unique().tolist()

        # Before stratified splitting, try to stratify drugs and cells
        # cell_stratified_list = []
        # for cell in copy.deepcopy(cell_list):
        #     try:
        #         cell_stratified_list.append(cl_meta_df['site'].loc[cell])
        #     except KeyError:
        #         logger.warning('Removing cell %s '
        #                        'for not included in metadata' % cell)
        #         cell_list.remove(cell)

        drug_stratified_list = []
        for drug in drug_list:
            drug_stratified_list.append(drug_analysis_df.loc[drug].values)

        # Change validation size when both features are disjoint in splitting
        if self.__disjoint_cells and self.__disjoint_drugs:
            self.__validation_size = self.__validation_size ** 0.7

        # Randomly take the validation_size portion of drugs/cells
        training_cell_list, validation_cell_list = \
            train_test_split(cell_list,
                             test_size=self.__validation_size,
                             random_state=self.__rand_state,
                             # stratify=cell_stratified_list,
                             shuffle=True)
        training_drug_list, validation_drug_list = \
            train_test_split(drug_list,
                             test_size=self.__validation_size,
                             random_state=self.__rand_state,
                             stratify=drug_stratified_list,
                             shuffle=True)

        logger.debug('Split training/validation data '
                     '(disjoint_cells=%r, disjoint_drugs=%r) ...'
                     % (self.__disjoint_cells, self.__disjoint_drugs))

        # Split data based on disjoint cell/drug strategy
        if self.__disjoint_cells and self.__disjoint_drugs:

            training_drug_resp_df = self.__drug_resp_df.loc[
                (self.__drug_resp_df['CELLNAME'].isin(training_cell_list)) &
                (self.__drug_resp_df['DRUG_ID'].isin(training_drug_list))]

            validation_drug_resp_df = self.__drug_resp_df.loc[
                (self.__drug_resp_df['CELLNAME'].isin(validation_cell_list)) &
                (self.__drug_resp_df['DRUG_ID'].isin(validation_drug_list))]

        elif self.__disjoint_cells and (not self.__disjoint_drugs):

            training_drug_resp_df = self.__drug_resp_df.loc[
                self.__drug_resp_df['CELLNAME'].isin(training_cell_list)]

            validation_drug_resp_df = self.__drug_resp_df.loc[
                self.__drug_resp_df['CELLNAME'].isin(validation_cell_list)]

        elif (not self.__disjoint_cells) and self.__disjoint_drugs:

            training_drug_resp_df = self.__drug_resp_df.loc[
                self.__drug_resp_df['DRUG_ID'].isin(training_drug_list)]

            validation_drug_resp_df = self.__drug_resp_df.loc[
                self.__drug_resp_df['DRUG_ID'].isin(validation_drug_list)]

        else:
            training_drug_resp_df, validation_drug_resp_df = \
                train_test_split(self.__drug_resp_df,
                                 test_size=self.__validation_size,
                                 random_state=self.__rand_state,
                                 shuffle=False)

        # Make sure that if not disjoint, the training/validation set should
        #  share the same drugs/cells
        if not self.__disjoint_cells:
            # Make sure that cell lines are common
            common_cells = set(training_drug_resp_df['CELLNAME'].unique()) & \
                           set(validation_drug_resp_df['CELLNAME'].unique())

            training_drug_resp_df = training_drug_resp_df.loc[
                training_drug_resp_df['CELLNAME'].isin(common_cells)]
            validation_drug_resp_df = validation_drug_resp_df.loc[
                validation_drug_resp_df['CELLNAME'].isin(common_cells)]

        if not self.__disjoint_drugs:
            # Make sure that drugs are common
            common_drugs = set(training_drug_resp_df['DRUG_ID'].unique()) & \
                           set(validation_drug_resp_df['DRUG_ID'].unique())

            training_drug_resp_df = training_drug_resp_df.loc[
                training_drug_resp_df['DRUG_ID'].isin(common_drugs)]
            validation_drug_resp_df = validation_drug_resp_df.loc[
                validation_drug_resp_df['DRUG_ID'].isin(common_drugs)]

        validation_ratio = \
            len(validation_drug_resp_df) / len(training_drug_resp_df)
        if validation_ratio < 0.15 or validation_ratio > 0.3:
            logger.warning('Suspicious validation/training ratio: %.2f' %
                           validation_ratio)

        # return training_drug_resp_df or validation_drug_resp_df
        self.__drug_resp_df = training_drug_resp_df if self.training \
            else validation_drug_resp_df
        self.__trim_dataframes(trim_data_source=False)


# Test segment for drug response dataset
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    data_gen = DrugRespDataset(
        data_folder='../../data/',
        data_source='NCI60',
        disjoint_cells=True,
        disjoint_drugs=True,
        int_dtype=np.int32,
        float_dtype=np.float32,
        growth_scaling='none',
        training=True)

    num_operations = 10000
    start_time = time.time()
    for i in range(num_operations):
        tmp = data_gen[i]
    print('%i Get Operations Running Time: %.1f Seconds.'
          % (num_operations, (time.time() - start_time)))
