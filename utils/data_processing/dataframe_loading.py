""" 
    File Name:          UnoPytorch/dataframe_loading.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/30/18
    Python Version:     3.6.6
    File Description:   
        This file takes care of all the dataframe loading and basic
        pre-processing from raw data.
"""
import json
import multiprocessing
import os
import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats


from utils.data_processing.dataframe_scaling import scale_dataframe
from utils.data_processing.label_encoding import encode_label_to_int
from utils.miscellaneous.file_downloading import download_files

logger = logging.getLogger(__name__)

# Folders for raw/processed data
RAW_FOLDER = './raw/'
PROC_FOLDER = './processed/'

# All the filenames related to the project
DRUG_RESP_FILENAME = 'rescaled_combined_single_drug_growth'
ECFP_FILENAME = 'pan_drugs_dragon7_ECFP.tsv'
PFP_FILENAME = 'pan_drugs_dragon7_PFP.tsv'
DSCPTR_FILENAME = 'pan_drugs_dragon7_descriptors.tsv'
CL_METADATA = 'combined_cl_metadata'
RNASEQ_SOURCE_SCALE_FILENAME = 'combined_rnaseq_data_lincs1000_source_scale'
RNASEQ_COMBAT_FILENAME = 'combined_rnaseq_data_lincs1000_combat'


def get_all_drugs(data_root: str):

    file_path = os.path.join(data_root, PROC_FOLDER, 'drugs.txt')

    # If the list if drugs already exists, load and continue ##################
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    # Otherwise get common drugs from all the dataframes ######################
    logger.debug('Processing common drug lists ... ')

    # Download the raw file if not exist
    download_files(filenames=[DRUG_RESP_FILENAME,
                              ECFP_FILENAME,
                              PFP_FILENAME,
                              DSCPTR_FILENAME],
                   target_folder=os.path.join(data_root, RAW_FOLDER))

    # Only load the drug IDs for faster processing time
    resp_drugs = set(pd.read_csv(
        os.path.join(data_root, RAW_FOLDER, DRUG_RESP_FILENAME),
        sep='\t',
        header=0,
        index_col=None,
        usecols=['DRUG_ID', ]).values.flatten())

    ecfp_drugs = set(pd.read_csv(
        os.path.join(data_root, RAW_FOLDER, ECFP_FILENAME),
        sep='\t',
        header=None,
        usecols=[0, ],
        skiprows=[0, ]).values.flatten())

    pfp_drugs = set(pd.read_csv(
        os.path.join(data_root, RAW_FOLDER, PFP_FILENAME),
        sep='\t',
        header=None,
        usecols=[0, ],
        skiprows=[0, ]).values.flatten())

    dscptr_drgus = set(pd.read_csv(
        os.path.join(data_root, RAW_FOLDER, DSCPTR_FILENAME),
        sep='\t',
        header=0,
        usecols=[0, ],
        na_values='na').values.flatten())

    # Takes the common elements from all 4 sets of drugs
    drugs = list(resp_drugs & ecfp_drugs & pfp_drugs & dscptr_drgus)

    # save to disk for future usage
    try:
        os.makedirs(os.path.join(data_root, PROC_FOLDER))
    except FileExistsError:
        pass

    with open(file_path, 'w') as f:
        json.dump(drugs, f, indent=4)
    return drugs


def get_all_cells(data_root: str):

    file_path = os.path.join(data_root, PROC_FOLDER, 'cells.txt')

    # If the list if drugs already exists, load and continue ##################
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    # Otherwise get common drugs from all the dataframes ######################
    logger.debug('Processing common cell lines lists ... ')

    # Download the raw file if not exist
    download_files(filenames=[DRUG_RESP_FILENAME,
                              RNASEQ_COMBAT_FILENAME,
                              RNASEQ_SOURCE_SCALE_FILENAME],
                   target_folder=os.path.join(data_root, RAW_FOLDER))

    # Only load the cell lines for faster processing time
    resp_cells = set(pd.read_csv(
        os.path.join(data_root, RAW_FOLDER, DRUG_RESP_FILENAME),
        sep='\t',
        header=0,
        index_col=None,
        usecols=['CELLNAME', ]).values.flatten())

    combat_cells = set(pd.read_csv(
        os.path.join(data_root, RAW_FOLDER, RNASEQ_COMBAT_FILENAME),
        sep='\t',
        header=0,
        usecols=[0, ]).values.flatten())

    source_scale_cells = set(pd.read_csv(
        os.path.join(data_root, RAW_FOLDER, RNASEQ_SOURCE_SCALE_FILENAME),
        sep='\t',
        header=0,
        usecols=[0, ]).values.flatten())

    # Takes the common elements from all 3 sets of cell lines
    cells = list(resp_cells & combat_cells & source_scale_cells)

    # save to disk for future usage
    try:
        os.makedirs(os.path.join(data_root, PROC_FOLDER))
    except FileExistsError:
        pass

    with open(file_path, 'w') as f:
        json.dump(cells, f, indent=4)
    return cells


def get_drug_resp_df(
        data_root: str,

        grth_scaling: str,

        int_dtype: type = np.int8,
        float_dtype: type = np.float32):

    df_filename = 'drug_resp_df(scaling=%s).pkl' % grth_scaling
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        logger.debug('Processing drug response dataframe ... ')

        # Download the raw file if not exist
        download_files(filenames=DRUG_RESP_FILENAME,
                       target_folder=os.path.join(data_root, RAW_FOLDER))

        df = pd.read_csv(
            os.path.join(data_root, RAW_FOLDER, DRUG_RESP_FILENAME),
            sep='\t',
            header=0,
            index_col=None,
            usecols=[0, 1, 2, 4, 6, ])

        # Encode data sources into numeric
        data_src_dict_path = os.path.join(data_root, PROC_FOLDER,
                                          'data_src_dict.txt')
        df['SOURCE'] = encode_label_to_int(df['SOURCE'], data_src_dict_path)

        # Scaling the growth with given scaling method
        df['GROWTH'] = scale_dataframe(df['GROWTH'], grth_scaling)

        # Convert data type into generic python types
        df[['SOURCE']] = df[['SOURCE']].astype(int)
        df[['LOG_CONCENTRATION', 'GROWTH']] = \
            df[['LOG_CONCENTRATION', 'GROWTH']].astype(float)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    df[['SOURCE']] = df[['SOURCE']].astype(int_dtype)
    df[['LOG_CONCENTRATION', 'GROWTH']] = \
        df[['LOG_CONCENTRATION', 'GROWTH']].astype(float_dtype)
    return df


def get_drug_fgpt_df(
        data_root: str,

        int_dtype: type = np.int8):

    df_filename = 'drug_fgpt_df.pkl'
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        logger.debug('Processing drug fingerprint dataframe ... ')

        # Download the raw file if not exist
        download_files(filenames=[ECFP_FILENAME, PFP_FILENAME],
                       target_folder=os.path.join(data_root, RAW_FOLDER))

        ecfp_df = pd.read_csv(
            os.path.join(data_root, RAW_FOLDER, ECFP_FILENAME),
            sep='\t',
            header=None,
            index_col=0,
            skiprows=[0, ])

        pfp_df = pd.read_csv(
            os.path.join(data_root, RAW_FOLDER, PFP_FILENAME),
            sep='\t',
            header=None,
            index_col=0,
            skiprows=[0, ])

        df = pd.concat([ecfp_df, pfp_df], axis=1, join='inner')

        # Convert data type into generic python types
        df = df.astype(int)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    df = df.astype(int_dtype)
    return df


def get_drug_dscptr_df(
        data_root: str,

        dscptr_scaling: str,
        nan_thresh: float,

        float_dtype: type = np.float32):

    df_filename = 'drug_dscptr_df(scaling=%s, nan_thresh=%.2f).pkl' \
                  % (dscptr_scaling, nan_thresh)
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        logger.debug('Processing drug descriptor dataframe ... ')

        # Download the raw file if not exist
        download_files(filenames=DSCPTR_FILENAME,
                       target_folder=os.path.join(data_root, RAW_FOLDER))

        df = pd.read_csv(
            os.path.join(data_root, RAW_FOLDER, DSCPTR_FILENAME),
            sep='\t',
            header=0,
            index_col=0,
            na_values='na')

        # Drop NaN values if the percentage of NaN exceeds nan_threshold
        # Note that columns (features) are dropped first, and then rows (drugs)
        valid_thresh = 1.0 - nan_thresh

        df.dropna(axis=1, inplace=True, thresh=int(df.shape[0] * valid_thresh))
        df.dropna(axis=0, inplace=True, thresh=int(df.shape[1] * valid_thresh))

        # Fill the rest of NaN with column means
        df.fillna(df.mean(), inplace=True)

        # Scaling the descriptor with given scaling method
        df = scale_dataframe(df, dscptr_scaling)

        # Convert data type into generic python types
        df = df.astype(float)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    df = df.astype(float_dtype)
    return df


def get_drug_feature_df(
        data_root: str,

        feature_usage: str,
        dscptr_scaling: str,
        dscptr_nan_thresh: float,

        int_dtype: type = np.int8,
        float_dtype: type = np.float32):

    # Return the corresponding drug feature dataframe
    if feature_usage == 'both':
        return pd.concat([get_drug_fgpt_df(data_root=data_root,
                                           int_dtype=int_dtype),
                          get_drug_dscptr_df(data_root=data_root,
                                             dscptr_scaling=dscptr_scaling,
                                             nan_thresh=dscptr_nan_thresh,
                                             float_dtype=float_dtype)],
                         axis=1, join='inner')
    elif feature_usage == 'fingerprint':
        return get_drug_fgpt_df(data_root=data_root,
                                int_dtype=int_dtype)
    elif feature_usage == 'descriptor':
        return get_drug_dscptr_df(data_root=data_root,
                                  dscptr_scaling=dscptr_scaling,
                                  nan_thresh=dscptr_nan_thresh,
                                  float_dtype=float_dtype)
    else:
        logger.error('Drug feature must be one of \'fingerprint\', '
                     '\'descriptor\', or \'both\'.', exc_info=True)
        raise ValueError('Undefined drug feature %s.' % feature_usage)


def get_rna_seq_df(
        data_root: str,

        feature_usage: str,
        rnaseq_scaling: str,

        float_dtype: type = np.float32):

    df_filename = 'rnaseq_df(%s, scaling=%s).pkl' \
                  % (feature_usage, rnaseq_scaling)
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        logger.debug('Processing RNA sequence dataframe ... ')

        if feature_usage == 'source_scale':
            raw_data_filename = RNASEQ_SOURCE_SCALE_FILENAME
        elif feature_usage == 'combat':
            raw_data_filename = RNASEQ_COMBAT_FILENAME
        else:
            logger.error('Unknown RNA feature %s.' % feature_usage,
                         exc_info=True)
            raise ValueError('RNA feature usage must be one of '
                             '\'source_scale\' or \'combat\'.')

        # Download the raw file if not exist
        download_files(filenames=raw_data_filename,
                       target_folder=os.path.join(data_root, RAW_FOLDER))

        df = pd.read_csv(
            os.path.join(data_root, RAW_FOLDER, raw_data_filename),
            sep='\t',
            header=0,
            index_col=0)

        # Scaling the descriptor with given scaling method
        df = scale_dataframe(df, rnaseq_scaling)

        # Convert data type into generic python types
        df = df.astype(float)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    df = df.astype(float_dtype)
    return df


def get_combo_stats_df(
        data_root: str,

        grth_scaling: str,

        int_dtype: type = np.int8,
        float_dtype: type = np.float32):

    df_filename = 'combo_stats_df(scaling=%s).pkl' % grth_scaling
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise process combo statistics and save #############################
    else:
        logger.debug('Processing drug + cell combo statics dataframe ... '
                     'this may take up to 5 minutes.')

        # Load the whole drug response dataframe and create a combo column
        # Use generic python dtypes to minimize the error during processing
        drug_resp_df = get_drug_resp_df(data_root=data_root,
                                        grth_scaling=grth_scaling,
                                        int_dtype=int,
                                        float_dtype=float)

        # logger.debug('Limiting the dataframe with drugs and cell lines ... ')
        # drug_resp_df = drug_resp_df.loc[
        #     (drug_resp_df['CELLNAME'].isin(get_all_cells(data_root))) &
        #     (drug_resp_df['DRUG_ID'].isin(get_all_drugs(data_root)))]

        # Using a dict to store all combo info with a single iteration
        combo_dict = {}

        # Note that there are different ways of iterating the dataframe
        # Fastest way is to convert the dataframe into ndarray, which is
        # 2x faster than itertuples(), which is 110x faster than iterrows().
        # This part takes about 30 sec on AMD 2700X
        drug_resp_array = drug_resp_df.values

        # Each row in the drug response dataframe contains:
        # ['SOURCE', 'DRUG_ID', 'CELLNAME', 'LOG_CONCENTRATION', 'GROWTH']
        for row in drug_resp_array:

            # row[1] = drug
            # row[2] = cell
            # row[3] = concentration
            # row[4] = growth

            # The combo name is made of drug + cell line
            combo = row[1] + '+' + row[2]
            if combo not in combo_dict:
                # Each dictionary value will be a list containing:
                # [drug, cell, tuple of concentration, tuple of growth]
                combo_dict[combo] = [row[1], row[2], (), ()]

            # Concentration and growth
            combo_dict[combo][2] += (row[3], )
            combo_dict[combo][3] += (row[4], )

        # Using list of lists (table) for much faster data access
        # This part is parallelized using joblib
        def process_combo(dict_value: list):

            # Each dict value will be a list containing:
            # [drug, cell, tuple of concentration, tuple of growth]
            conc_tuple = dict_value[2]
            grth_tuple = dict_value[3]

            # This might throw warnings as var(growth) == 0 sometimes
            # Fill NaN with 0 as there is no correlation in cases like this
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    corr = stats.pearsonr(x=conc_tuple, y=grth_tuple)[0]
                except Warning:
                    corr = 0.0
            # corr = 0. if np.isnan(corr) else corr

            # Each row contains the following fields:
            # ['DRUG_ID', 'CELLNAME','NUM_REC', 'AVG', 'VAR', 'CORR']
            return [dict_value[0], dict_value[1], len(conc_tuple),
                    np.mean(grth_tuple), corr]

        num_cores = multiprocessing.cpu_count()
        combo_stats = Parallel(n_jobs=num_cores)(
            delayed(process_combo)(v) for _, v in combo_dict.items())

        # Convert ths list of lists to dataframe
        cols = ['DRUG_ID', 'CELLNAME', 'NUM_REC', 'AVG_GRTH', 'CORR']
        df = pd.DataFrame(combo_stats, columns=cols)

        # Convert data type into generic python types
        df[['NUM_REC']] = df[['NUM_REC']].astype(int)
        df[['AVG_GRTH', 'CORR']] = df[['AVG_GRTH', 'CORR']].astype(float)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    df[['NUM_REC']] = df[['NUM_REC']].astype(int_dtype)
    df[['AVG_GRTH', 'CORR']] = df[['AVG_GRTH', 'CORR']].astype(float_dtype)
    return df


def get_drug_stats_df(
        data_root: str,

        grth_scaling: str,

        int_dtype: type = np.int16,
        float_dtype: type = np.float32):

    if int_dtype == np.int8:
        logger.warning('Integer length too smaller for drug statistics.')

    df_filename = 'drug_stats_df(scaling=%s).pkl' % grth_scaling
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise process combo statistics and save #############################
    else:
        logger.debug('Processing drug statics dataframe ... ')

        # Load combo (drug + cell) dataframe to construct drug statistics
        combo_stats_df = get_combo_stats_df(data_root=data_root,
                                            grth_scaling=grth_scaling,
                                            int_dtype=int,
                                            float_dtype=float)

        # Using a dict to store all drug info with a single iteration
        drug_dict = {}

        # Each row in the combo stats dataframe contains:
        # ['DRUG_ID', 'CELLNAME','NUM_REC', 'AVG_GRTH', 'CORR']
        combo_stats_array = combo_stats_df.values
        for row in combo_stats_array:
            drug = row[0]
            if drug not in drug_dict:
                # Each dictionary value will be a list containing:
                # [num of cell, tuple of num of records,
                #  tuple of avg grth, tuple of corr]
                drug_dict[drug] = [0, (), (), ()]

            drug_dict[drug][0] += 1
            drug_dict[drug][1] += (row[2], )
            drug_dict[drug][2] += (row[3], )
            drug_dict[drug][3] += (row[4], )

        # Using list of lists (table) for much faster data access
        # This part is parallelized using joblib
        def process_drug(drug: str, dict_value: list):

            # Each row in the drug stats dataframe contains:
            # ['DRUG_ID', 'NUM_CL', 'NUM_REC', 'AVG_GRTH', 'AVG_CORR']
            num_cl = dict_value[0]
            records_tuple = dict_value[1]

            assert num_cl == len(records_tuple)

            grth_tuple = dict_value[2]
            corr_tuple = dict_value[3]

            num_rec = np.sum(records_tuple)
            avg_grth = np.average(a=grth_tuple, weights=records_tuple)
            avg_corr = np.average(a=corr_tuple, weights=records_tuple)

            return [drug, num_cl, num_rec, avg_grth, avg_corr]

        num_cores = multiprocessing.cpu_count()
        drug_stats = Parallel(n_jobs=num_cores)(
            delayed(process_drug)(k, v) for k, v in drug_dict.items())

        # Convert ths list of lists to dataframe
        cols = ['DRUG_ID', 'NUM_CL', 'NUM_REC', 'AVG_GRTH', 'AVG_CORR']
        df = pd.DataFrame(drug_stats, columns=cols)

        # Convert data type into generic python types
        df[['NUM_CL', 'NUM_REC']] = df[['NUM_CL', 'NUM_REC']].astype(int)
        df[['AVG_GRTH', 'AVG_CORR']] = \
            df[['AVG_GRTH', 'AVG_CORR']].astype(float)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    df[['NUM_CL', 'NUM_REC']] = df[['NUM_CL', 'NUM_REC']].astype(int_dtype)
    df[['AVG_GRTH', 'AVG_CORR']] = \
        df[['AVG_GRTH', 'AVG_CORR']].astype(float_dtype)
    return df


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    # Test the cell/drug list functions
    print('=' * 80 + '\n'
          'In drug response dataframes (growth, drug feature, RNA sequence), '
          'there are %i unique drugs and %i unique cell lines.'
          % (len(get_all_drugs(data_root='../../data/')),
             len(get_all_cells(data_root='../../data/'))))

    # Test all basic data loading functions
    print('=' * 80 + '\nDrug response dataframe head:')
    print(get_drug_resp_df(data_root='../../data/',
                           grth_scaling='none').head())

    print('=' * 80 + '\nDrug feature dataframe head:')
    print(get_drug_feature_df(data_root='../../data/',
                              feature_usage='both',
                              dscptr_scaling='std',
                              dscptr_nan_thresh=0.).head())

    print('=' * 80 + '\nRNA sequence dataframe head:')
    print(get_rna_seq_df(data_root='../../data/',
                         feature_usage='source_scale',
                         rnaseq_scaling='std').head())

    # Test statistic data loading functions
    print('=' * 80 + '\nDrug statistics dataframe head:')
    print(get_drug_stats_df(data_root='../../data/',
                            grth_scaling='none').head())
