"""
    File Name:          UnoPytorch/random_seeding.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/29/18
    Python Version:     3.6.6
    File Description:

"""
import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_processing.cell_line_dataframes import get_cl_meta_df

STD_SCALE = 0.1
IMAGE_SIZE = (24, 16)


def load_result_file(trn_src: str,
                     val_src: str,

                     results_dir: str,
                     epoch: int = None,
                     early_stop_patience: int = 5):

    # If epoch number is not given or invalid, retrieve the best epoch using
    #  the early stop patience.
    if epoch is None or epoch <= 0:

        # https://docs.python.org/3/library/glob.html#glob.glob
        file_name_pattern = \
            '[[]trn=%s[]][[]val=%s[]][[]epoch=[0-9][0-9][]].csv' \
            % (trn_src, val_src)
        file_names = glob.glob(os.path.join(results_dir, file_name_pattern))

        epochs = sorted([re.findall(r'\d+', fn)[-1] for fn in file_names])
        epoch = int(epochs[-(early_stop_patience + 1)])

        file_name = '[trn=%s][val=%s][epoch=%02i].csv' \
                    % (trn_src, val_src, epoch)

    # If epoch is given, make sure that the corresponding file exists
    else:

        file_name = '[trn=%s][val=%s][epoch=%02i].csv' \
                    % (trn_src, val_src, epoch)

        if not os.path.exists(os.path.join(results_dir, file_name)):
            raise FileNotFoundError('No file named %s in given dir.'
                                    % file_name)

    # Load CSV file and process into numpy array
    file_path = os.path.join(results_dir, file_name)
    result_array = pd.read_csv(file_path).values

    # Result array have the following columns:
    # [drug_id, cell_id, concentration, growth, prediction, *uq]
    uq_predictions = result_array[:, 5:]
    result_array = result_array[:, :5]

    mae = np.abs(np.subtract(result_array[:, -1],
                             result_array[:, -2])).reshape(-1, 1)
    mse = np.square(mae).reshape(-1, 1)
    uq = np.std(np.array(uq_predictions, dtype=np.float64),
                axis=1).reshape(-1, 1)

    # Returned pd dataframe has the structure of
    # [drug_id, cell_id, concentration, growth, prediction, mse, mae, uq]
    ret = np.concatenate((result_array, mse, mae, uq), axis=1)
    ret = pd.DataFrame(ret,
                       columns=['drug_id', 'cell_id', 'concentration',
                                'growth', 'prediction', 'mse', 'mae', 'uq'])
    return epoch, ret


def plot_error_bar_over_uq(num_bars: int,

                           trn_src: str,
                           val_src: str,

                           results_dir: str,
                           epoch: int = None,
                           early_stop_patience: int = 5,

                           error_type: str = 'mae',
                           equal_partition: bool = True,
                           image_dir: str = '../../results/images/'):

    # This function plots MSE/MAE over log(UQ)

    if error_type.lower() not in ['mse', 'mae']:
        raise ValueError('Error type must be \'MSE\' or \'MSE\'')

    # Load result file
    epoch, results = load_result_file(
        trn_src, val_src, results_dir, epoch, early_stop_patience)

    uq_array = results['uq'].values.flatten()
    error_array = results[error_type.lower()].values.flatten()

    # Plot the averaged error based on UQ
    bar_labels = []
    uq_indicators = []
    avg_error_in_partition = []
    scaled_std_error_in_partition = []

    # In this case, each bar has the same number of samples
    if equal_partition:

        ordered_indices = np.argsort(uq_array)
        partition_size = int(np.floor(len(uq_array) / num_bars))

        for i in range(num_bars):
            start_index = i * partition_size
            end_index = (i + 1) * partition_size

            partitioned_indices = \
                ordered_indices[start_index: end_index]

            partitioned_error_array = \
                error_array[partitioned_indices]

            uq_indicators.append('[%.1f, %.1f)'
                                % (uq_array[ordered_indices[start_index]],
                                   uq_array[ordered_indices[end_index - 1]]))

            bar_labels.append('n=%i' % len(partitioned_error_array))
            avg_error_in_partition.append(np.mean(partitioned_error_array))
            scaled_std_error_in_partition.append(
                np.std(partitioned_error_array) * STD_SCALE)

    # In this case, each bar has the same incremental UQ
    else:

        max_uq, min_uq = np.max(uq_array), np.min(uq_array)
        start = min_uq
        step = (max_uq - min_uq) / num_bars

        for i in range(num_bars):
            uq_indicators.append('[%.1f, %.1f)'
                                % (start, (start + step)))

            partitioned_error_array = \
                error_array[(uq_array >= start) & (uq_array < start + step)]

            bar_labels.append('n=%i' % len(partitioned_error_array))
            avg_error_in_partition.append(np.mean(partitioned_error_array))
            scaled_std_error_in_partition.append(
                np.std(partitioned_error_array) * STD_SCALE)

            start += step

    # Figure size and text
    plt.figure(figsize=IMAGE_SIZE)
    plt.xlabel('Uncertainty Quantification '
               '(STD of %i Predictions with MC Dropout)' % len(uq_array))
    plt.ylabel('Averaged %s' % error_type.upper())
    plt.title('Averaged Error over Uncertainty '
              '(Trained on %s and Validated on %s, Epoch %i)'
              % (trn_src, val_src, epoch))

    # Labeling each bar with the scaled std and the number of samples
    bars = plt.bar(uq_indicators, avg_error_in_partition,
                   yerr=scaled_std_error_in_partition, align='center',
                   alpha=0.5, ecolor='black', capsize=4)
    for bar, label in zip(bars, bar_labels):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                 label, ha='center', va='bottom')

    plt.show()

    # TODO: Save the plot into image folder

    return


def plot_error_bar_over_cell(cl_class: str,

                             trn_src: str,
                             val_src: str,

                             results_dir: str,
                             epoch: int = None,
                             early_stop_patience: int = 5,

                             error_type: str = 'mse',
                             image_dir: str = '../../results/images/'):

    # This function plots error (MSE/MAE) and UQ over cell types
    # This is going to be a bar plot
    cl_class = cl_class.lower()
    if cl_class not in ['site', 'type', 'category']:
        raise ValueError('Cell line class must be one of '
                         '\'site\', \'type\', or \'category\'.')

    if error_type.lower() not in ['mse', 'mae']:
        raise ValueError('Error type must be \'MSE\' or \'MSE\'')

    # Load result file
    epoch, results = load_result_file(
        trn_src, val_src, results_dir, epoch, early_stop_patience)

    # Get the (un-encoded) cell line metadata for classification
    cl_meta_df = get_cl_meta_df('../../data/', encoding=False)
    cl_classes = cl_meta_df[cl_class].unique()

    bar_labels = []
    # class_indicators = []
    avg_error_in_class = []
    scaled_std_error_in_class = []

    for c in cl_classes:

        # Get all the cell lines that are in this classes
        cl_in_class = cl_meta_df.loc[cl_meta_df[cl_class] == c].index.tolist()

        error_array_in_class = results.loc[results['cell_id'].isin(
            cl_in_class)][error_type.lower()].values.flatten()

        bar_labels.append('n=%i' % len(error_array_in_class))

        if len(error_array_in_class) != 0:
            avg_error_in_class.append(np.mean(error_array_in_class))
            scaled_std_error_in_class.append(
                np.std(error_array_in_class) * STD_SCALE)
        else:
            avg_error_in_class.append(0.)
            scaled_std_error_in_class.append(0.)

    plt.figure(figsize=IMAGE_SIZE)
    plt.xlabel('Cell Line Classes')
    plt.ylabel('Averaged %s' % error_type.upper())
    plt.title('Averaged Error over Cell Line %s '
              '(Trained on %s and Validated on %s, Epoch %i)'
              % (cl_class, trn_src, val_src, epoch))

    # Labeling each bar with the scaled std and the number of samples
    bars = plt.bar(cl_classes, avg_error_in_class,
                   yerr=scaled_std_error_in_class, align='center',
                   alpha=0.5, ecolor='black', capsize=4)
    plt.xticks(cl_classes, rotation=-75)
    for bar, label in zip(bars, bar_labels):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                 label, ha='center', va='bottom')

    plt.show()


    return


def plot_metric_over_uq_cutoff(
        trn_src: str,
        val_src: str,
        epoch: int,
        uq_cutoff: float,
        metric_type: str = 'mse'):
    # This function plots metric (average MSE/MAE or R2) over UQ cutoff.
    # For example, average MAE over top 5% UQ predictions

    return


# Test segment for results analysis
if __name__ == '__main__':

    # Test basic data loading function
    # result = load_result_file(
    #     trn_src='NCI60',
    #     val_src='CTRP',
    #     results_dir='../../results/saved_predictions(1021_1006)')

    # Test error versus uncertainty bar plot
    # data_srcs = ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI']
    # for trn_src in data_srcs:
    #     for val_src in data_srcs:
    #         plot_error_bar_over_uq(
    #             num_bars=20,
    #
    #             trn_src=trn_src,
    #             val_src=val_src,
    #             results_dir='../../results/saved_predictions(1021_1006)')

    # Test error versus cell line classes bar plot
    data_srcs = ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI']
    for trn_src in data_srcs:
        for val_src in data_srcs:
            plot_error_bar_over_cell(
                cl_class='type',

                trn_src=trn_src,
                val_src=val_src,
                results_dir='../../results/saved_predictions(1021_1006)')
