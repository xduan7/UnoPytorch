""" 
    File Name:          UnoPytorch/encoder_init.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/16/18
    Python Version:     3.6.6
    File Description:   

"""
import copy
import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR

from networks.encoder_net import EncoderNet
from utils.datasets.basic_dataset import DataFrameDataset
from utils.network_config.optimizer import get_optimizer
from utils.miscellaneous.random_seeding import seed_random_state


def get_autoencoder(
        model_path: str,
        df_path: str,

        # Autoencoder network configuration
        layer_dim: int,
        latent_dim: int,
        num_layers: int,

        # Major training parameters
        ae_loss_func: str = 'mse',
        ae_opt: str = 'sgd',
        ae_lr: float = 1e-3,

        # Unimportant training parameters
        validation_ratio: float = 0.2,
        trn_batch_size: int = 32,
        val_batch_size: int = 256,
        max_num_epochs: int = 100,
        early_stop_patience: int = 10,
        decay_factor: float = 1.00,

        # Miscellaneous
        device: torch.device = torch.device(type='cuda'),
        verbose: bool = True,
        rand_state: int = 0, ):

    # Check if the model exists, load and return
    if os.path.exists(model_path):
        return torch.load(model_path)

    if verbose:
        print('Constructing autoencoders ... ')

    # Load dataframe, split and construct dataloaders #########################
    trn_df, val_df = train_test_split(pd.read_pickle(df_path),
                                      test_size=validation_ratio,
                                      random_state=rand_state,
                                      shuffle=True)
    dataloader_kwargs = {
        'shuffle': 'True',
        # 'num_workers': multiprocessing.cpu_count() if use_cuda else 0,
        'num_workers': 0,
        'pin_memory': True if device == torch.device('cuda') else False
    }

    trn_dataloader = torch.utils.data.DataLoader(
        DataFrameDataset(trn_df),
        batch_size=trn_batch_size,
        **dataloader_kwargs)

    val_dataloader = torch.utils.data.DataLoader(
        DataFrameDataset(val_df),
        batch_size=val_batch_size,
        **dataloader_kwargs)

    # Start training and until converge #######################################
    











    # Store the best AE and return the best one ###############################
    pass




def encoder_init(
        ae_int: bool,
        model_path: str,
        training_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        rand_state: int,

        layer_dim: int,
        latent_dim: int,
        num_layers: int,

        precision: str = 'full',
        loss_func: str = 'mse',
        optimizer: str = 'sgd',
        learning_rate: float = 1e-3,
        decay_factor: float = 1.00,
        early_stop_patience: int = 10,
        training_batch_size: int = 32,
        validation_batch_size: int = 1024,
        num_epochs: int = 100,

        device: str = 'cuda',
        verbose: bool = True, ):

    # If autoencoder initialization is not required, return a fresh encoder
    if not ae_int:
        net = EncoderNet(
            input_dim=training_df.shape[1],
            layer_dim=layer_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            autoencoder=False).to(device)
        return net.half() if precision.lower() == 'half' else net

    # If the model exists, load and return
    if os.path.exists(model_path):
        net = torch.load(model_path)
        return net.half() if precision.lower() == 'half' else net

    # Setting up random seed for reproducible and deterministic results
    seed_random_state(rand_state)

    # Otherwise, construct and train a new autoencoder
    loader_kwargs = \
        {'num_workers': multiprocessing.cpu_count(),
         'pin_memory': True} if device == 'cuda' else {}

    training_dataloader = torch.utils.data.DataLoader(
        DataFrameDataset(training_df),
        batch_size=training_batch_size,
        shuffle=True,
        **loader_kwargs)

    validation_dataloader = torch.utils.data.DataLoader(
        DataFrameDataset(validation_df),
        batch_size=validation_batch_size,
        shuffle=True,
        **loader_kwargs)

    net = EncoderNet(
        input_dim=training_df.shape[1],
        layer_dim=layer_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        autoencoder=True).to(device)
    net = net.half() if precision.lower() == 'half' else net

    opt = get_optimizer(opt_type=optimizer,
                        networks=net,
                        learning_rate=learning_rate)
    lr_decay = LambdaLR(opt, lr_lambda=lambda e: decay_factor ** e)


    loss_func = F.l1_loss if loss_func == 'l1' else F.mse_loss

    # Training and validation loop
    best_val_loss = np.inf
    best_model = None
    patience = 0
    for epoch in range(1, num_epochs + 1):

        lr_decay.step(epoch - 1)

        # Training
        net.train()
        train_loss = 0.
        for batch_idx, feature in enumerate(training_dataloader):

            feature = feature.to(device).half() if precision == 'half' \
                else feature.to(device)
            net.zero_grad()
            pred = net(feature)

            loss = loss_func(pred, feature)
            train_loss += loss.item() * len(feature)
            loss.backward()
            opt.step()
        train_loss /= len(training_dataloader.dataset)

        # Validation
        net.eval()
        val_loss = 0.
        with torch.no_grad():
            for feature in validation_dataloader:

                feature = feature.to(device).half() if precision == 'half' \
                    else feature.to(device)
                pred = net(feature)
                loss = loss_func(pred, feature)
                val_loss += loss.item() * len(feature)

            val_loss /= len(validation_dataloader.dataset)

        if verbose:
            print('Epoch %4i: training loss: %.4f;\t validation loss: %.4f'
                  % (epoch, train_loss, val_loss))

        if val_loss < best_val_loss:
            patience = 0
            best_val_loss = val_loss
            best_model = copy.deepcopy(net)

        else:
            patience += 1
            if patience > early_stop_patience:
                if verbose:
                    print('Evoking early stopping. Best validation loss %.4f.'
                          % best_val_loss)
                break

    torch.save(best_model, model_path)
    return best_model


def get_gene_encoder(
        model_folder: str,

        training_df: pd.DataFrame,
        validation_df: pd.DataFrame,

        layer_dim: int,
        latent_dim: int,
        num_layers: int,

        rnaseq_feature_usage: str,
        rnaseq_scaling: str,

        encoder_training_kwarg: dict):

    model_name = 'gene_net(%i*%i=>%i, %s, scaling=%s)' \
                 % (layer_dim, num_layers, latent_dim,
                    rnaseq_feature_usage, rnaseq_scaling)
    model_path = os.path.join(model_folder, model_name)

    return encoder_init(
        model_path=model_path,
        training_df=training_df,
        validation_df=validation_df,

        layer_dim=layer_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,

        **encoder_training_kwarg)


def get_drug_encoder(
        model_folder: str,

        training_df: pd.DataFrame,
        validation_df: pd.DataFrame,

        layer_dim: int,
        latent_dim: int,
        num_layers: int,

        descriptor_scaling: str,
        nan_threshold: float,
        drug_feature_usage: str,

        encoder_training_kwarg: dict):

    model_name = 'drug_net(%i*%i=>%i, %s, ' \
                 'descriptor_scaling=%s, nan_thresh=%.2f)' \
                 % (layer_dim, num_layers, latent_dim,
                    drug_feature_usage, descriptor_scaling, nan_threshold,)
    model_path = os.path.join(model_folder, model_name)

    return encoder_init(
        model_path=model_path,
        training_df=training_df,
        validation_df=validation_df,

        layer_dim=layer_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,

        **encoder_training_kwarg)


def get_encoders(

        data_folder: str,
        model_folder: str,

        autoencoder_init: bool,
        precision: str,

        gene_layer_dim: int,
        gene_latent_dim: int,
        gene_num_layers: int,

        rnaseq_feature_usage: str,
        rnaseq_scaling: str,

        drug_layer_dim: int,
        drug_latent_dim: int,
        drug_num_layers: int,

        descriptor_scaling: str,
        nan_threshold: float,
        drug_feature_usage: str,

        rand_state: int = 0,
        verbose: bool = False):

    data_folder = os.path.join(data_folder, './processed/')

    # Load gene dataframe
    rnaseq_df_filename = 'rnaseq_df(%s, scaling=%s, dtype=float16).pkl' \
                         % (rnaseq_feature_usage, rnaseq_scaling)
    rnaseq_df_path = os.path.join(data_folder, rnaseq_df_filename)
    rnaseq_df = pd.read_pickle(rnaseq_df_path)
    trn_rnaseq_df, val_rnaseq_df = \
        train_test_split(rnaseq_df,
                         test_size=0.2,
                         random_state=rand_state,
                         shuffle=True)

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
        drug_feature_df = pd.concat([drug_fingerprint_df, drug_descriptor_df],
                                    axis=1, join='inner')
    elif drug_feature_usage == 'fingerprint':
        drug_feature_df = drug_fingerprint_df
    elif drug_feature_usage == 'descriptor':
        drug_feature_df = drug_descriptor_df
    else:
        raise ValueError('Undefined drug feature %s.' % drug_feature_usage)
    trn_drug_df, val_drug_df = \
        train_test_split(drug_feature_df,
                         test_size=0.2,
                         random_state=rand_state,
                         shuffle=True)

    encoder_training_kwarg = {
        'ae_int': autoencoder_init,
        'rand_state': rand_state,
        'precision': precision,

        'optimizer': 'Adam',
        'learning_rate': 5e-4,
        'early_stop_patience': 32,
        'num_epochs': 5000,

        'verbose': verbose}

    gene_encoder = get_gene_encoder(
        model_folder=model_folder,

        training_df=trn_rnaseq_df,
        validation_df=val_rnaseq_df,

        layer_dim=gene_layer_dim,
        latent_dim=gene_latent_dim,
        num_layers=gene_num_layers,

        rnaseq_feature_usage=rnaseq_feature_usage,
        rnaseq_scaling=rnaseq_scaling,

        encoder_training_kwarg=encoder_training_kwarg)

    drug_encoder = get_drug_encoder(
        model_folder=model_folder,

        training_df=trn_drug_df,
        validation_df=val_drug_df,

        layer_dim=drug_layer_dim,
        latent_dim=drug_latent_dim,
        num_layers=drug_num_layers,

        descriptor_scaling=descriptor_scaling,
        nan_threshold=nan_threshold,
        drug_feature_usage=drug_feature_usage,

        encoder_training_kwarg=encoder_training_kwarg)

    return gene_encoder, drug_encoder


if __name__ == '__main__':

    gene_encoder, drug_encoder = get_encoders(
        data_folder='../../data/',
        model_folder='../../models',

        autoencoder_init=True,
        precision='full',

        gene_layer_dim=1024,
        gene_latent_dim=512,
        gene_num_layers=2,

        rnaseq_feature_usage='combat',
        rnaseq_scaling='std',

        drug_layer_dim=4096,
        drug_latent_dim=1024,
        drug_num_layers=2,

        descriptor_scaling='std',
        nan_threshold=0.,
        drug_feature_usage='both',

        rand_state=0,
        verbose=True, )

    print(gene_encoder)
    print(drug_encoder)
