""" 
    File Name:          UnoPytorch/uno_pytorch.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:   

"""

import argparse
import json
import multiprocessing

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.optim import Adam, RMSprop, SGD
from torch.optim.lr_scheduler import LambdaLR

from networks.resp_net import RespNet
from utils.datasets.drug_resp_dataset import DrugRespDataset
from utils.miscellaneous.dataframe_scaling import SCALING_METHODS
from utils.miscellaneous.encoder_init import get_encoders


def main():
    # Training settings and hyper-parameters
    parser = argparse.ArgumentParser(
        description='Multitasking Neural Network for Genes and Drugs')

    # Dataset parameters (data sources, scaling, feature usages, etc.)
    parser.add_argument('--training_src', type=str, required=True,
                        help='source of drug response for training')
    parser.add_argument('--validation_src', type=str, required=True, nargs='+',
                        help='list of sources of drug response for validation')
    parser.add_argument('--precision', type=str, default='full',
                        help='neural network and dataset precision',
                        choices=['full', 'half', ])

    parser.add_argument('--growth_scaling', type=str, default='std',
                        help='scaling method for drug response (growth)',
                        choices=SCALING_METHODS)
    parser.add_argument('--descriptor_scaling', type=str, default='std',
                        help='scaling method for drug feature (descriptor)',
                        choices=SCALING_METHODS)
    parser.add_argument('--rnaseq_scaling', type=str, default='std',
                        help='scaling method for RNA sequence',
                        choices=SCALING_METHODS)
    parser.add_argument('--nan_threshold', type=float, default=0.0,
                        help='ratio of NaN values allowed for drug features')

    parser.add_argument('--rnaseq_feature_usage', type=str, default='combat',
                        help='ratio of NaN values allowed for drug features',
                        choices=['source_scale', 'combat', ])
    parser.add_argument('--drug_feature_usage', type=str, default='both',
                        help='drug features (fp and/or desc) used',
                        choices=['fingerprint', 'descriptor', 'both', ])
    parser.add_argument('--validation_size', type=float, default=0.2,
                        help='ratio for validation dataset')
    parser.add_argument('--disjoint_drugs', action='store_true',
                        help='disjoint drugs between train/validation')
    parser.add_argument('--disjoint_cells', action='store_true',
                        help='disjoint cells between train/validation')

    # Network architecture and dimensions
    parser.add_argument('--autoencoder_init', action='store_true',
                        help='indicator of autoencoder initialization')

    parser.add_argument('--gene_layer_dim', type=int, default=1024,
                        help='dimension of layers for RNA sequence')
    parser.add_argument('--gene_latent_dim', type=int, default=256,
                        help='dimension of latent variable for RNA sequence')
    parser.add_argument('--gene_num_layers', type=int, default=2,
                        help='number of layers for RNA sequence')

    parser.add_argument('--drug_layer_dim', type=int, default=4096,
                        help='dimension of layers for drug feature')
    parser.add_argument('--drug_latent_dim', type=int, default=1024,
                        help='dimension of latent variable for drug feature')
    parser.add_argument('--drug_num_layers', type=int, default=2,
                        help='number of layers for drug feature')

    parser.add_argument('--resp_layer_dim', type=int, default=1024,
                        help='dimension of layers for drug response block')
    parser.add_argument('--resp_num_layers', type=int, default=2,
                        help='number of layers for drug response block')
    parser.add_argument('--resp_dropout', type=float, default=0.2,
                        help='dropout of residual blocks for drug response')
    parser.add_argument('--resp_num_blocks', type=int, default=3,
                        help='number of residual blocks for drug response')

    parser.add_argument('--resp_activation', type=str, default='none',
                        help='activation for response prediction output')

    # Multitasking

    # Training parameters
    parser.add_argument('--loss_func', type=str, default='mse',
                        help='loss function for training')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer type for training')
    parser.add_argument('--resp_lr', type=float, default=1e-5,
                        help='drug response optimizer learning rate')
    parser.add_argument('--decay_factor', type=float, default=0.95,
                        help='decay factor for learning rate')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='patience/tolerance for early stopping')
    parser.add_argument('--training_batch_size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--validation_batch_size', type=int, default=256,
                        help='input batch size for validation')
    parser.add_argument('--max_num_batches', type=int, default=1000,
                        help='maximum number of epochs to train')
    parser.add_argument('--max_num_epochs', type=int, default=100,
                        help='maximum number of epochs to train')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='logging interval for training status.')

    # Miscellaneous
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--rand_state', type=int, default=0,
                        help='random state of numpy/sklearn/pytorch')

    args = parser.parse_args()
    print('Training Arguments:\n' + json.dumps(vars(args), indent=4))

    # Setting up random seed first
    np.random.seed(args.rand_state)
    torch.manual_seed(args.rand_state)
    torch.cuda.manual_seed_all(args.rand_state)

    # Computation device config
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    num_workers = multiprocessing.cpu_count()

    # Data loaders for training/validation ####################################
    drug_resp_dataset_kwargs = {
        'data_folder': './data/',
        'rand_state': args.rand_state,
        'summary': False,

        'int_dtype': np.int8,
        'float_dtype': np.float16,
        'output_dtype': np.float16 if args.precision == 'half' else np.float32,

        'growth_scaling': args.growth_scaling,
        'descriptor_scaling': args.descriptor_scaling,
        'rnaseq_scaling': args.rnaseq_scaling,
        'nan_threshold': args.nan_threshold,

        'rnaseq_feature_usage': args.rnaseq_feature_usage,
        'drug_feature_usage': args.drug_feature_usage,
        'validation_size': args.validation_size,
        'disjoint_drugs': args.disjoint_drugs,
        'disjoint_cells': args.disjoint_cells, }
    dataloader_kwargs = {
        'shuffle': 'True',
        'num_workers': num_workers if use_cuda else 0,
        'pin_memory': True if use_cuda else False, }

    drug_resp_trn_loader = torch.utils.data.DataLoader(
        DrugRespDataset(data_source=args.training_src,
                        training=True,
                        **drug_resp_dataset_kwargs),
        batch_size=args.training_batch_size,
        **dataloader_kwargs)

    # List of data loaders for different validation sets
    drug_resp_val_loaders = [torch.utils.data.DataLoader(
        DrugRespDataset(data_source=src,
                        training=False,
                        **drug_resp_dataset_kwargs),
        batch_size=args.validation_batch_size,
        **dataloader_kwargs) for src in args.validation_src]

    # Constructing and initializing neural networks ###########################
    gene_encoder, drug_encoder = get_encoders(
        data_folder='./data/',
        model_folder='./models/',

        autoencoder_init=args.autoencoder_init,
        precision=args.precision,

        gene_layer_dim=args.gene_layer_dim,
        gene_latent_dim=args.gene_latent_dim,
        gene_num_layers=args.gene_num_layers,

        rnaseq_feature_usage=args.rnaseq_feature_usage,
        rnaseq_scaling=args.rnaseq_scaling,

        drug_layer_dim=args.drug_layer_dim,
        drug_latent_dim=args.drug_latent_dim,
        drug_num_layers=args.drug_num_layers,

        descriptor_scaling=args.descriptor_scaling,
        nan_threshold=args.nan_threshold,
        drug_feature_usage=args.drug_feature_usage,

        rand_state=args.rand_state,
        verbose=False)

    resp_net = RespNet(
        gene_latent_dim=args.gene_latent_dim,
        drug_latent_dim=args.drug_latent_dim,

        gene_encoder=gene_encoder.encoder,
        drug_encoder=drug_encoder.encoder,

        resp_layer_dim=args.resp_layer_dim,
        resp_num_layers=args.resp_num_layers,
        resp_dropout=args.resp_dropout,
        resp_num_blocks=args.resp_num_blocks,

        resp_activation=args.resp_activation).to(device)

    resp_net = nn.DataParallel(resp_net.half()) if args.precision == 'half' \
        else nn.DataParallel(resp_net)

    if args.optimizer.lower() == 'adam':
        opt = Adam(resp_net.parameters(), lr=args.resp_lr, amsgrad=True)
    elif args.optimizer.lower() == 'rmsprop':
        opt = RMSprop(resp_net.parameters(), lr=args.resp_lr, momentum=0.2)
    else:
        opt = SGD(resp_net.parameters(), lr=args.resp_lr, momentum=0.8)

    lr_decay = LambdaLR(opt, lr_lambda=lambda e: args.decay_factor ** e)
    loss_func = F.l1_loss if args.loss_func == 'l1' else F.mse_loss

    num_batches = np.amin((len(drug_resp_trn_loader), args.max_num_batches))
    log_interval = int(num_batches / args.log_interval)

    val_mse, val_mae, val_r2 = [], [], []
    best_r2 = -np.inf
    patience = 0
    start_time = time.time()

    for epoch in range(1, args.max_num_epochs + 1):

        epoch_start_time = time.time()
        lr_decay.step(epoch - 1)

        print('=' * 80)
        resp_net.train()
        for batch_idx, (rnaseq, drug_feature, conc, growth) \
                in enumerate(drug_resp_trn_loader):

            if batch_idx >= num_batches:
                break

            rnaseq, drug_feature, conc, growth = \
                rnaseq.to(device), drug_feature.to(device), \
                conc.to(device), growth.to(device)
            resp_net.zero_grad()

            pred = resp_net(rnaseq, drug_feature, conc)
            loss = loss_func(pred, growth)
            loss.backward()
            opt.step()

            if batch_idx % log_interval == 0:
                print('Training Epoch %3d (%04.1f%%)\t\tLoss: %10.4f' % (
                    epoch, 100. * batch_idx / num_batches, loss.item()))

        print('\nValidation Results: ')
        resp_net.eval()
        with torch.no_grad():
            for val_loader in drug_resp_val_loaders:

                mse, mae = 0., 0.
                growth_array, pred_array = np.array([]), np.array([])

                for rnaseq, drug_feature, conc, growth in val_loader:
                    rnaseq, drug_feature, conc, growth = \
                        rnaseq.to(device), drug_feature.to(device), \
                        conc.to(device), growth.to(device)
                    pred = resp_net(rnaseq, drug_feature, conc)

                    num_samples = conc.shape[0]
                    mse += F.mse_loss(pred, growth).item() * num_samples
                    mae += F.l1_loss(pred, growth).item() * num_samples

                    growth_array = np.concatenate(
                        (growth_array, growth.cpu().numpy().flatten()))
                    pred_array = np.concatenate(
                        (pred_array, pred.cpu().numpy().flatten()))

                mse /= len(val_loader.dataset)
                mae /= len(val_loader.dataset)
                r2 = r2_score(y_pred=pred_array, y_true=growth_array)

                print('\t%8s \t MSE: %8.2f \t MAE: %8.2f \t R2: %+4.2f' %
                      (val_loader.dataset.data_source, mse, mae, r2))

                val_mse.append(mse)
                val_mae.append(mae)
                val_r2.append(r2)

                if val_loader.dataset.data_source == args.training_src:
                    if r2 > best_r2:
                        patience = 0
                        best_r2 = r2
                    else:
                        patience += 1

            if patience >= args.early_stop_patience:
                print('Validation results does not improve for %d epochs ... '
                      'invoking early stopping.' % patience)
                break

        print('Epoch Running Time: %.1f Seconds.'
              % (time.time() - epoch_start_time))

    val_mse, val_mae, val_r2 = \
        np.array(val_mse).reshape(-1, len(args.validation_src)), \
        np.array(val_mae).reshape(-1, len(args.validation_src)), \
        np.array(val_r2).reshape(-1, len(args.validation_src))

    # print(val_mse)
    # print(val_mae)
    # print(val_r2)

    print('Program Running Time: %.1f Seconds.' % (time.time() - start_time))

    # Print overall validation results
    val_data_sources = \
        [loader.dataset.data_source for loader in drug_resp_val_loaders]

    best_r2_scores = np.amax(val_r2, axis=0)
    best_epochs = np.argmax(val_r2, axis=0)

    print('=' * 80)
    print('Overall Validation Results:')
    for index, data_source in enumerate(val_data_sources):
        print('\t%6s \t Best R2 Score: %+6.4f '
              '(Epoch = %3d, MSE = %8.2f, MAE = %8.2f)'
              % (data_source, best_r2_scores[index],
                 best_epochs[index] + 1,
                 val_mse[best_epochs[index], index],
                 val_mae[best_epochs[index], index]))


main()
