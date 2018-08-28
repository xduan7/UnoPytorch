""" 
    File Name:          UnoPytorch/encoder_init.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/16/18
    Python Version:     3.6.6
    File Description:   

"""
import logging
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from networks.encoder_net import EncoderNet
from utils.datasets.basic_dataset import DataFrameDataset
from utils.network_config.optimizer import get_optimizer
from utils.miscellaneous.random_seeding import seed_random_state

logger = logging.getLogger(__name__)


def get_encoder(
        model_path: str,
        dataframe: pd.DataFrame,

        # Autoencoder network configuration
        autoencoder_init: bool,
        layer_dim: int,
        num_layers: int,
        latent_dim: int,

        # Major training parameters
        ae_loss_func: str,
        ae_opt: str,
        ae_lr: float,
        lr_decay_factor: float,
        max_num_epochs: int,
        early_stop_patience: int,

        # Secondary training parameters
        validation_ratio: float = 0.2,
        trn_batch_size: int = 32,
        val_batch_size: int = 1024,

        # Miscellaneous
        device: torch.device = torch.device('cuda'),
        verbose: bool = True,
        rand_state: int = 0, ):
    """encoder = gene_encoder = get_gene_encoder(./models/', dataframe,
           True, 1000, 3, 100, 'mse', 'sgd', 1e-3, 0.98, 100, 10)

    This function constructs, initializes and returns a feature encoder for
    the given dataframe.

    When parameter autoencoder_init is set to False, it simply construct and
    return an encoder with simple initialization (nn.init.xavier_normal_).

    When autoencoder_init is set to True, the function will return the
    encoder part of an autoencoder trained on the given data. It will first
    check if the model file exists. If not, it will start training with
    given training hyper-parameters.

    Note that the saved model in disk contains the whole autoencoder (
    encoder and decoder). But the function only returns the encoder.

    """

    # If autoencoder initialization is not required, return a plain encoder
    if not autoencoder_init:
        return EncoderNet(input_dim=dataframe.shape[1],
                          layer_dim=layer_dim,
                          num_layers=num_layers,
                          latent_dim=latent_dim,
                          autoencoder=False).to(device).encoder

    # Check if the model exists, load and return
    if os.path.exists(model_path):
        logger.debug('Loading existing autoencoder model from %s ...'
                     % model_path)
        return torch.load(model_path).encoder

    logger.debug('Constructing autoencoder from dataframe ...')

    # Setting up random seed for reproducible and deterministic results
    seed_random_state(rand_state)

    # Load dataframe, split and construct dataloaders #########################
    trn_df, val_df = train_test_split(dataframe,
                                      test_size=validation_ratio,
                                      random_state=rand_state,
                                      shuffle=True)
    dataloader_kwargs = {
        'shuffle': 'True',
        # 'num_workers': multiprocessing.cpu_count() if use_cuda else 0,
        'pin_memory': True if device == torch.device('cuda') else False, }

    trn_dataloader = DataLoader(DataFrameDataset(trn_df),
                                batch_size=trn_batch_size,
                                **dataloader_kwargs)

    val_dataloader = DataLoader(DataFrameDataset(val_df),
                                batch_size=val_batch_size,
                                **dataloader_kwargs)

    # Construct the network and get prepared for training #####################
    autoencoder = EncoderNet(input_dim=dataframe.shape[1],
                             layer_dim=layer_dim,
                             latent_dim=latent_dim,
                             num_layers=num_layers,
                             autoencoder=True).to(device)
    assert ae_loss_func.lower() == 'l1' or ae_loss_func.lower() == 'mse'
    loss_func = F.l1_loss if ae_loss_func.lower() == 'l1' else F.mse_loss

    optimizer = get_optimizer(opt_type=ae_opt,
                              networks=autoencoder,
                              learning_rate=ae_lr)
    lr_decay = LambdaLR(optimizer, lr_lambda=lambda e: lr_decay_factor ** e)

    # Train until max number of epochs is reached or early stopped ############
    best_val_loss = np.inf
    best_autoencoder = None
    patience = 0

    if verbose:
        print('=' * 80)
        print('Training log for autoencoder model (%s): ' % model_path)

    for epoch in range(max_num_epochs):

        lr_decay.step(epoch)

        # Training loop for autoencoder
        autoencoder.train()
        trn_loss = 0.
        for batch_idx, samples in enumerate(trn_dataloader):
            samples = samples.to(device)
            recon_samples = autoencoder(samples)
            autoencoder.zero_grad()

            loss = loss_func(input=recon_samples, target=samples)
            loss.backward()
            optimizer.step()

            trn_loss += loss.item() * len(samples)
        trn_loss /= len(trn_dataloader.dataset)

        # Validation loop for autoencoder
        autoencoder.eval()
        val_loss = 0.
        with torch.no_grad():
            for samples in val_dataloader:
                samples = samples.to(device)
                recon_samples = autoencoder(samples)
                loss = loss_func(input=recon_samples, target=samples)

                val_loss += loss.item() * len(samples)
            val_loss /= len(val_dataloader.dataset)

        if verbose:
            print('Epoch %4i: training loss: %.4f;\t validation loss: %.4f'
                  % (epoch + 1, trn_loss, val_loss))

        # Save the model to memory if it achieves best validation loss
        if val_loss < best_val_loss:
            patience = 0
            best_val_loss = val_loss
            best_autoencoder = copy.deepcopy(autoencoder)
        # Otherwise increase patience and check for early stopping
        else:
            patience += 1
            if patience > early_stop_patience:
                if verbose:
                    print('Evoking early stopping. Best validation loss %.4f.'
                          % best_val_loss)
                break

    # Store the best autoencoder and return it ################################
    try:
        os.makedirs(os.path.dirname(model_path))
    except FileExistsError:
        pass
    torch.save(best_autoencoder, model_path)
    return best_autoencoder.encoder


def get_gene_encoder(
        model_folder: str,
        data_folder: str,

        # RNA sequence usage and scaling
        rnaseq_feature_usage: str,
        rnaseq_scaling: str,

        # Autoencoder network configuration
        autoencoder_init: bool,
        layer_dim: int,
        num_layers: int,
        latent_dim: int,

        # Training keyword parameters to be provided
        training_kwarg: dict,

        # Miscellaneous
        device: torch.device = torch.device('cuda'),
        verbose: bool = True,
        rand_state: int = 0, ):

    gene_encoder_name = 'gene_net(%i*%i=>%i, %s, scaling=%s).pt' % \
                        (layer_dim, num_layers, latent_dim,
                         rnaseq_feature_usage, rnaseq_scaling)
    gene_encoder_path = os.path.join(model_folder, gene_encoder_name)

    gene_df_name = 'rnaseq_df(%s, scaling=%s, dtype=float16).pkl' \
                   % (rnaseq_feature_usage, rnaseq_scaling)
    gene_df_path = os.path.join(data_folder, 'processed', gene_df_name)
    gene_df = pd.read_pickle(gene_df_path)

    return get_encoder(
        model_path=gene_encoder_path,
        dataframe=gene_df,

        autoencoder_init=autoencoder_init,
        layer_dim=layer_dim,
        num_layers=num_layers,
        latent_dim=latent_dim,

        **training_kwarg,

        device=device,
        verbose=verbose,
        rand_state=rand_state, )


def get_drug_encoder(
        model_folder: str,
        data_folder: str,

        # Drug feature usage and scaling
        drug_feature_usage: str,
        descriptor_scaling: str,
        nan_threshold: float,

        # Autoencoder network configuration
        autoencoder_init: bool,
        layer_dim: int,
        num_layers: int,
        latent_dim: int,

        # Training keyword parameters to be provided
        training_kwarg: dict,

        # Miscellaneous
        device: torch.device = torch.device('cuda'),
        verbose: bool = True,
        rand_state: int = 0, ):

    drug_encoder_name = 'drug_net(%i*%i=>%i, %s, descriptor_scaling=%s, ' \
                        'nan_thresh=%.2f).pt' % \
        (layer_dim, num_layers, latent_dim,
         drug_feature_usage, descriptor_scaling, nan_threshold,)
    drug_encoder_path = os.path.join(model_folder, drug_encoder_name)

    # Load drug dataframe
    drug_fingerprint_df_filename = 'drug_fingerprint_df(dtype=int8).pkl'
    drug_fingerprint_df_path = os.path.join(data_folder,
                                            drug_fingerprint_df_filename)
    drug_fingerprint_df = pd.read_pickle(drug_fingerprint_df_path)

    drug_descriptor_df_filename = \
        'drug_descriptor_df(scaling=%s, nan_thresh=%.2f, dtype=float16).pkl' \
        % (descriptor_scaling, nan_threshold)
    drug_descriptor_df_path = os.path.join(data_folder,
                                           drug_descriptor_df_filename)
    drug_descriptor_df = pd.read_pickle(drug_descriptor_df_path)

    if drug_feature_usage == 'both':
        drug_df = pd.concat([drug_fingerprint_df, drug_descriptor_df],
                            axis=1, join='inner')
    elif drug_feature_usage == 'fingerprint':
        drug_df = drug_fingerprint_df
    elif drug_feature_usage == 'descriptor':
        drug_df = drug_descriptor_df
    else:
        raise ValueError('Undefined drug feature %s.' % drug_feature_usage)

    return get_encoder(
        model_path=drug_encoder_path,
        dataframe=drug_df,

        autoencoder_init=autoencoder_init,
        layer_dim=layer_dim,
        num_layers=num_layers,
        latent_dim=latent_dim,

        **training_kwarg,

        device=device,
        verbose=verbose,
        rand_state=rand_state, )


if __name__ == '__main__':

    # Test code for autoencoder with RNA sequence and drug features
    ae_training_kwarg = {
        'ae_loss_func': 'mse',
        'ae_opt': 'adam',
        'ae_lr': 1e-3,
        'lr_decay_factor': 0.95,
        'max_num_epochs': 10,
        'early_stop_patience': 50, }

    gene_encoder = get_gene_encoder(
        model_folder='../../models/',
        data_folder='../../data/',

        rnaseq_feature_usage='source_scale',
        rnaseq_scaling='std',

        autoencoder_init=True,
        layer_dim=1024,
        num_layers=2,
        latent_dim=256,

        training_kwarg=ae_training_kwarg,

        device=torch.device('cuda'),
        verbose=True,
        rand_state=0, )

    drug_encoder = get_drug_encoder(
        model_folder='../../models/',
        data_folder='../../data/',

        drug_feature_usage='both',
        descriptor_scaling='std',
        nan_threshold=0.0,

        autoencoder_init=True,
        layer_dim=1024,
        num_layers=2,
        latent_dim=256,

        training_kwarg=ae_training_kwarg,

        device=torch.device('cuda'),
        verbose=True,
        rand_state=0, )
