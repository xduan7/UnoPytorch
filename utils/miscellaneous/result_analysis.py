"""
    File Name:          UnoPytorch/random_seeding.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/29/18
    Python Version:     3.6.6
    File Description:

"""

import numpy as np
import matplotlib.pyplot as plt

def load_results_file(
        trn_src: str,
        val_src: str,
        epoch: int):



    # Results are in the format of [drug, cell, target, pred, mse, mae, uq] (np.ndarray)
    return None


def plot_error_over_uq(
        trn_src: str,
        val_src: str,
        epoch: int,
        error_type: str = 'mse',
        image_dir: str = '../../results/images/'):

    # This function plots MSE/MAE over log(UQ)
    # This is going to be a scatter plot



    if error_type.lower() not in ['mse', 'mae']:
        raise ValueError('Error type must be \'MSE\' or \'MSE\'')


    # Load result file
    results = load_results_file(trn_src, val_src, epoch)

    uq = results[:, -1]

    if error_type.lower() == 'mse':
        error = results[:, -3]
    else:
        error = results[:, -2]

    plt.scatter(uq, error)
    plt.title('%s versus UQ (training_src=%s, validation_src=%s, epoch=%i)'
              % (error_type.upper(), trn_src, val_src, epoch))
    plt.show()

    # Save the plot into image folder

    return




def plot_error_over_cell(
        trn_src: str,
        val_src: str,
        epoch: int,
        error_type: str = 'mse'):

    # This function plots error (MSE/MAE) and UQ over cell types
    # This is going to be a bar plot


    if error_type.lower() not in ['mse', 'mae']:
        raise ValueError('Error type must be \'MSE\' or \'MSE\'')

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