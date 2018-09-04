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
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import LambdaLR

from networks.classification_net import ClfNet
from networks.response_net import RespNet
from utils.data_processing.label_encoding import get_label_dict
from utils.datasets.drug_resp_dataset import DrugRespDataset
from utils.datasets.cl_class_dataset import CLClassDataset
from utils.data_processing.dataframe_scaling import SCALING_METHODS
from utils.network_config.encoder_init import get_gene_encoder, \
    get_drug_encoder
from utils.network_config.optimizer import get_optimizer
from utils.miscellaneous.random_seeding import seed_random_state


# Number of workers for dataloader. Too many workers might lead to process
# hanging for pytorch version 4.1. Set this number between 0 and 4.
NUM_WORKER = 4

DATA_ROOT = './data/'


def train_resp(
        device: torch.device,

        resp_net: nn.Module,
        data_loader: torch.utils.data.DataLoader,

        max_num_batches: int,
        loss_func: callable,
        optimizer: torch.optim, ):

    resp_net.train()
    total_loss = 0.
    num_samples = 0

    for batch_idx, (rnaseq, drug_feature, conc, grth) \
            in enumerate(data_loader):

        if batch_idx >= max_num_batches:
            break

        rnaseq, drug_feature, conc, grth = \
            rnaseq.to(device), drug_feature.to(device), \
            conc.to(device), grth.to(device)
        resp_net.zero_grad()

        pred_growth = resp_net(rnaseq, drug_feature, conc)
        loss = loss_func(pred_growth, grth)
        loss.backward()
        optimizer.step()

        num_samples += conc.shape[0]
        total_loss += loss.item() * conc.shape[0]

    print('\tDrug Response Regression Loss: %8.2f'
          % (total_loss / num_samples))


def valid_resp(
        device: torch.device,

        resp_net: nn.Module,
        data_loaders: iter, ):

    resp_net.eval()

    mse_list = []
    mae_list = []
    r2_list = []

    print('\tDrug Response Regression:')

    with torch.no_grad():
        for val_loader in data_loaders:

            mse, mae = 0., 0.
            growth_array, pred_array = np.array([]), np.array([])

            for rnaseq, drug_feature, conc, grth in val_loader:
                rnaseq, drug_feature, conc, grth = \
                    rnaseq.to(device), drug_feature.to(device), \
                    conc.to(device), grth.to(device)
                pred_growth = resp_net(rnaseq, drug_feature, conc)

                num_samples = conc.shape[0]
                mse += F.mse_loss(pred_growth, grth).item() * num_samples
                mae += F.l1_loss(pred_growth, grth).item() * num_samples

                growth_array = np.concatenate(
                    (growth_array, grth.cpu().numpy().flatten()))
                pred_array = np.concatenate(
                    (pred_array, pred_growth.cpu().numpy().flatten()))

            mse /= len(val_loader.dataset)
            mae /= len(val_loader.dataset)
            r2 = r2_score(y_pred=pred_array, y_true=growth_array)

            mse_list.append(mse)
            mae_list.append(mae)
            r2_list.append(r2)

            print('\t\t%-6s \t MSE: %8.2f \t MAE: %8.2f \t R2: %+4.2f' %
                  (val_loader.dataset.data_source, mse, mae, r2))

    return mse_list, mae_list, r2_list


def train_clf(
        device: torch.device,

        category_clf_net: nn.Module,
        site_clf_net: nn.Module,
        type_clf_net: nn.Module,
        data_loader: torch.utils.data.DataLoader,

        max_num_batches: int,
        optimizer: torch.optim, ):

    category_clf_net.train()
    site_clf_net.train()
    type_clf_net.train()

    for batch_idx, (rnaseq, data_src, cl_site, cl_type, cl_category) \
            in enumerate(data_loader):

        if batch_idx >= max_num_batches:
            break

        rnaseq, data_src, cl_site, cl_type, cl_category = \
            rnaseq.to(device), data_src.to(device), cl_site.to(device), \
            cl_type.to(device), cl_category.to(device)

        category_clf_net.zero_grad()
        site_clf_net.zero_grad()
        type_clf_net.zero_grad()

        out_category = category_clf_net(rnaseq, data_src)
        out_site = site_clf_net(rnaseq, data_src)
        out_type = type_clf_net(rnaseq, data_src)

        F.nll_loss(out_category, cl_category).backward()
        F.nll_loss(out_site, cl_site).backward()
        F.nll_loss(out_type, cl_type).backward()

        optimizer.step()


def valid_clf(
        device: torch.device,

        category_clf_net: nn.Module,
        site_clf_net: nn.Module,
        type_clf_net: nn.Module,
        data_loader: torch.utils.data.DataLoader, ):

    category_clf_net.eval()
    site_clf_net.eval()
    type_clf_net.eval()

    correct_category = 0
    correct_site = 0
    correct_type = 0

    with torch.no_grad():
        for rnaseq, data_src, cl_site, cl_type, cl_category in data_loader:

            rnaseq, data_src, cl_site, cl_type, cl_category = \
                rnaseq.to(device), data_src.to(device), cl_site.to(device), \
                cl_type.to(device), cl_category.to(device)

            out_category = category_clf_net(rnaseq, data_src)
            out_site = site_clf_net(rnaseq, data_src)
            out_type = type_clf_net(rnaseq, data_src)

            pred_category = out_category.max(1, keepdim=True)[1]
            pred_site = out_site.max(1, keepdim=True)[1]
            pred_type = out_type.max(1, keepdim=True)[1]

            correct_category += pred_category.eq(
                cl_category.view_as(pred_category)).sum().item()
            correct_site += pred_site.eq(
                cl_site.view_as(pred_site)).sum().item()
            correct_type += pred_type.eq(
                cl_type.view_as(pred_type)).sum().item()

    # Get overall accuracy
    category_acc = 100. * correct_category / len(data_loader.dataset)
    site_acc = 100. * correct_site / len(data_loader.dataset)
    type_acc = 100. * correct_type / len(data_loader.dataset)

    print('\tCell Line Classification: '
          '\n\t\tCategory Accuracy: \t\t%5.2f%%; '
          '\n\t\tSite Accuracy: \t\t\t%5.2f%%; '
          '\n\t\tType Accuracy: \t\t\t%5.2f%%'
          % (category_acc, site_acc, type_acc))

    return category_acc, site_acc, type_acc


def main():
    # Training settings and hyper-parameters
    parser = argparse.ArgumentParser(
        description='Multitasking Neural Network for Genes and Drugs')

    # Dataset parameters ######################################################
    # Training and validation data sources
    parser.add_argument('--trn_src', type=str, required=True,
                        help='training source for drug response')
    parser.add_argument('--val_srcs', type=str, required=True, nargs='+',
                        help='validation list of sources for drug response')

    # Pre-processing for dataframes
    parser.add_argument('--growth_scaling', type=str, default='std',
                        help='scaling method for drug response (growth)',
                        choices=SCALING_METHODS)
    parser.add_argument('--descriptor_scaling', type=str, default='std',
                        help='scaling method for drug feature (descriptor)',
                        choices=SCALING_METHODS)
    parser.add_argument('--rnaseq_scaling', type=str, default='std',
                        help='scaling method for RNA sequence',
                        choices=SCALING_METHODS)
    parser.add_argument('--descriptor_nan_threshold', type=float, default=0.0,
                        help='ratio of NaN values allowed for drug descriptor')

    # Feature usage and partitioning settings
    parser.add_argument('--rnaseq_feature_usage', type=str, default='combat',
                        help='RNA sequence data used',
                        choices=['source_scale', 'combat', ])
    parser.add_argument('--drug_feature_usage', type=str, default='both',
                        help='drug features (fp and/or desc) used',
                        choices=['fingerprint', 'descriptor', 'both', ])
    parser.add_argument('--validation_ratio', type=float, default=0.2,
                        help='ratio for validation dataset')
    parser.add_argument('--disjoint_drugs', action='store_true',
                        help='disjoint drugs between train/validation')
    parser.add_argument('--disjoint_cells', action='store_true',
                        help='disjoint cells between train/validation')

    # Network configuration ###################################################
    # Encoders for drug features and RNA sequence (LINCS 1000)
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

    # Using autoencoder for drug/sequence encoder initialization
    parser.add_argument('--autoencoder_init', action='store_true',
                        help='indicator of autoencoder initialization for '
                             'drug/RNA sequence feature encoder')

    # Drug response regression network
    parser.add_argument('--resp_layer_dim', type=int, default=1024,
                        help='dimension of layers for drug response block')
    parser.add_argument('--resp_num_layers_per_block', type=int, default=2,
                        help='number of layers for drug response res block')
    parser.add_argument('--resp_num_blocks', type=int, default=3,
                        help='number of residual blocks for drug response')
    parser.add_argument('--resp_num_layers', type=int, default=2,
                        help='number of layers for drug response')
    parser.add_argument('--resp_dropout', type=float, default=0.2,
                        help='dropout of residual blocks for drug response')
    parser.add_argument('--resp_activation', type=str, default='none',
                        help='activation for response prediction output',
                        choices=['sigmoid', 'tanh', 'none'])

    # RNA sequence classification network(s)
    parser.add_argument('--clf_layer_dim', type=int, default=256,
                        help='dimension of layers for sequence classification')
    parser.add_argument('--clf_num_layers', type=int, default=1,
                        help='number of layers for sequence classification')

    # Training and validation parameters ######################################
    # Drug response regression training parameters
    parser.add_argument('--resp_loss_func', type=str, default='mse',
                        help='loss function for drug response training',
                        choices=['mse', 'l1'])
    parser.add_argument('--resp_opt', type=str, default='SGD',
                        help='optimizer for drug response training',
                        choices=['SGD', 'RMSprop', 'Adam'])
    parser.add_argument('--resp_lr', type=float, default=1e-5,
                        help='learning rate for drug response training')

    # Starting epoch for drug response validation
    parser.add_argument('--resp_val_start_epoch', type=int, default=0,
                        help='starting epoch for drug response validation')

    # Early stopping based on R2 score of drug response prediction
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='patience for early stopping based on drug '
                             'response validation R2 scores ')

    # RNA sequence classification training parameters
    parser.add_argument('--clf_opt', type=str, default='SGD',
                        help='optimizer for sequence classification '
                             'training',
                        choices=['SGD', 'RMSprop', 'Adam'])
    parser.add_argument('--clf_lr', type=float, default=1e-5,
                        help='learning rate for sequence classification '
                             'training')

    # Global/shared training parameters
    parser.add_argument('--lr_decay_factor', type=float, default=0.95,
                        help='decay factor for learning rate')
    parser.add_argument('--trn_batch_size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help='input batch size for validation')
    parser.add_argument('--max_num_batches', type=int, default=1000,
                        help='maximum number of batches per epoch')
    parser.add_argument('--max_num_epochs', type=int, default=100,
                        help='maximum number of epochs')

    # Miscellaneous settings ##################################################
    parser.add_argument('--multi_gpu', action='store_true', default=False,
                        help='enables multiple GPU process')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--rand_state', type=int, default=0,
                        help='random state of numpy/sklearn/pytorch')

    args = parser.parse_args()
    print('Training Arguments:\n' + json.dumps(vars(args), indent=4))

    # Setting up random seed for reproducible and deterministic results
    seed_random_state(args.rand_state)

    # Computation device config (cuda or cpu)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Data loaders for training/validation ####################################
    dataloader_kwargs = {
        'timeout': 1,
        'shuffle': 'True',
        # 'num_workers': multiprocessing.cpu_count() if use_cuda else 0,
        'num_workers': NUM_WORKER if use_cuda else 0,
        'pin_memory': True if use_cuda else False, }

    # Drug response dataloaders for training/validation
    drug_resp_dataset_kwargs = {
        'data_root': DATA_ROOT,
        'rand_state': args.rand_state,
        'summary': False,

        'int_dtype': np.int8,
        'float_dtype': np.float16,
        'output_dtype': np.float32,

        'grth_scaling': args.growth_scaling,
        'dscptr_scaling': args.descriptor_scaling,
        'rnaseq_scaling': args.rnaseq_scaling,
        'dscptr_nan_threshold': args.descriptor_nan_threshold,

        'rnaseq_feature_usage': args.rnaseq_feature_usage,
        'drug_feature_usage': args.drug_feature_usage,
        'validation_ratio': args.validation_ratio,
        'disjoint_drugs': args.disjoint_drugs,
        'disjoint_cells': args.disjoint_cells, }

    drug_resp_trn_loader = torch.utils.data.DataLoader(
        DrugRespDataset(data_src=args.trn_src,
                        training=True,
                        **drug_resp_dataset_kwargs),
        batch_size=args.trn_batch_size,
        **dataloader_kwargs)

    # List of data loaders for different validation sets
    drug_resp_val_loaders = [torch.utils.data.DataLoader(
        DrugRespDataset(data_src=src,
                        training=False,
                        **drug_resp_dataset_kwargs),
        batch_size=args.val_batch_size,
        **dataloader_kwargs) for src in args.val_srcs]

    # Cell line classification dataloaders for training/validation
    cl_class_dataset_kwargs = {
        'data_root': DATA_ROOT,
        'rand_state': args.rand_state,
        'summary': False,

        'int_dtype': np.int8,
        'float_dtype': np.float16,
        'output_dtype': np.float32,

        'rnaseq_scaling': args.rnaseq_scaling,

        'rnaseq_feature_usage': args.rnaseq_feature_usage,
        'validation_ratio': args.validation_ratio, }

    cl_class_trn_loader = torch.utils.data.DataLoader(
        CLClassDataset(training=True,
                       **cl_class_dataset_kwargs),
        batch_size=args.trn_batch_size,
        **dataloader_kwargs)

    cl_class_val_loader = torch.utils.data.DataLoader(
        CLClassDataset(training=False,
                       **cl_class_dataset_kwargs),
        batch_size=args.val_batch_size,
        **dataloader_kwargs)

    # Constructing and initializing neural networks ###########################
    # Autoencoder training hyper-parameters
    ae_training_kwarg = {
        'ae_loss_func': 'mse',
        'ae_opt': 'sgd',
        'ae_lr': 2e-1,
        'lr_decay_factor': 1.0,
        'max_num_epochs': 1000,
        'early_stop_patience': 50, }

    encoder_kwarg = {
        'model_folder': './models/',
        'data_root': DATA_ROOT,

        'autoencoder_init': args.autoencoder_init,
        'training_kwarg': ae_training_kwarg,

        'device': device,
        'verbose': True,
        'rand_state': args.rand_state, }

    # Get RNA sequence encoder
    gene_encoder = get_gene_encoder(
        rnaseq_feature_usage=args.rnaseq_feature_usage,
        rnaseq_scaling=args.rnaseq_scaling,

        layer_dim=args.gene_layer_dim,
        num_layers=args.gene_num_layers,
        latent_dim=args.gene_latent_dim,
        **encoder_kwarg)

    # Get drug feature encoder
    drug_encoder = get_drug_encoder(
        drug_feature_usage=args.drug_feature_usage,
        descriptor_scaling=args.descriptor_scaling,
        nan_threshold=args.descriptor_nan_threshold,

        layer_dim=args.drug_layer_dim,
        num_layers=args.drug_num_layers,
        latent_dim=args.drug_latent_dim,
        **encoder_kwarg)

    # Regressor for drug response
    resp_net = RespNet(
        gene_latent_dim=args.gene_latent_dim,
        drug_latent_dim=args.drug_latent_dim,

        gene_encoder=gene_encoder,
        drug_encoder=drug_encoder,

        resp_layer_dim=args.resp_layer_dim,
        resp_num_layers_per_block=args.resp_num_layers_per_block,
        resp_num_blocks=args.resp_num_blocks,
        resp_num_layers=args.resp_num_layers,
        resp_dropout=args.resp_dropout,

        resp_activation=args.resp_activation).to(device)

    print(resp_net)

    # Sequence classifier for category, site, and type
    clf_net_kwargs = {
        'encoder': gene_encoder,
        'condition_dim': len(get_label_dict(DATA_ROOT, 'data_src_dict.txt')),
        'latent_dim': args.gene_latent_dim,
        'layer_dim': args.clf_layer_dim,
        'num_layers': args.clf_num_layers, }

    category_clf_net = ClfNet(
        num_classes=len(get_label_dict(DATA_ROOT, 'category_dict.txt')),
        **clf_net_kwargs).to(device)
    site_clf_net = ClfNet(
        num_classes=len(get_label_dict(DATA_ROOT, 'site_dict.txt')),
        **clf_net_kwargs).to(device)
    type_clf_net = ClfNet(
        num_classes=len(get_label_dict(DATA_ROOT, 'type_dict.txt')),
        **clf_net_kwargs).to(device)

    # Multi-GPU settings
    if args.multi_gpu:
        resp_net = nn.DataParallel(resp_net)
        category_clf_net = nn.DataParallel(category_clf_net)
        site_clf_net = nn.DataParallel(site_clf_net)
        type_clf_net = nn.DataParallel(type_clf_net)

    # Optimizers, learning rate decay, and miscellaneous ######################
    resp_opt = get_optimizer(opt_type=args.resp_opt,
                             networks=resp_net,
                             learning_rate=args.resp_lr)
    clf_opt = get_optimizer(opt_type=args.clf_opt,
                            networks=[category_clf_net,
                                      site_clf_net,
                                      type_clf_net],
                            learning_rate=args.clf_lr)

    resp_lr_decay = LambdaLR(optimizer=resp_opt,
                             lr_lambda=lambda e: args.lr_decay_factor ** e)
    clf_lr_decay = LambdaLR(optimizer=clf_opt,
                            lr_lambda=lambda e: args.lr_decay_factor ** e)

    resp_loss_func = F.l1_loss if args.resp_loss_func == 'l1' else F.mse_loss

    resp_max_num_batches = np.amin(
        (len(drug_resp_trn_loader), args.max_num_batches))
    clf_max_num_batches = np.amin(
        (len(cl_class_trn_loader), args.max_num_batches))

    # Training/validation loops ###############################################
    val_acc, val_mse, val_mae, val_r2 = [], [], [], []
    best_r2 = -np.inf
    patience = 0
    start_time = time.time()

    # Index of validation dataloader with the same data source
    # as the training dataloader
    val_index = 0
    for idx, loader in enumerate(drug_resp_val_loaders):
        if loader.dataset.data_source == args.trn_src:
            val_index = idx

    for epoch in range(1, args.max_num_epochs + 1):

        print('=' * 80 + '\nTraining Epoch %3i:' % epoch)
        epoch_start_time = time.time()
        resp_lr_decay.step(epoch)
        clf_lr_decay.step(epoch)

        # Training cell line classifier
        train_clf(device=device,
                  category_clf_net=category_clf_net,
                  site_clf_net=site_clf_net,
                  type_clf_net=type_clf_net,
                  data_loader=cl_class_trn_loader,
                  max_num_batches=clf_max_num_batches,
                  optimizer=clf_opt)

        # Training drug response regressor
        train_resp(device=device,
                   resp_net=resp_net,
                   data_loader=drug_resp_trn_loader,
                   max_num_batches=resp_max_num_batches,
                   loss_func=resp_loss_func,
                   optimizer=resp_opt)

        print('\nValidation Results:')

        if epoch >= args.resp_val_start_epoch:

            # Validating cell line classifier
            category_acc, site_acc, type_acc = \
                valid_clf(device=device,
                          category_clf_net=category_clf_net,
                          site_clf_net=site_clf_net,
                          type_clf_net=type_clf_net,
                          data_loader=cl_class_val_loader, )
            val_acc.append([category_acc, site_acc, type_acc])

            # Validating drug response regressor
            mse, mae, r2 = valid_resp(device=device,
                                      resp_net=resp_net,
                                      data_loaders=drug_resp_val_loaders)

            # Save the validation results in nested list
            val_mse.append(mse)
            val_mae.append(mae)
            val_r2.append(r2)

            # Record the best R2 score (same data source)
            # and check for early stopping if no improvement for epochs
            if r2[val_index] > best_r2:
                patience = 0
                best_r2 = r2[val_index]
            else:
                patience += 1
            if patience >= args.early_stop_patience:
                print('Validation results does not improve for %d epochs ... '
                      'invoking early stopping.' % patience)
                break

        print('Epoch Running Time: %.1f Seconds.'
              % (time.time() - epoch_start_time))

    val_acc = np.array(val_acc).reshape(-1, 3)
    val_mse, val_mae, val_r2 = \
        np.array(val_mse).reshape(-1, len(args.val_srcs)), \
        np.array(val_mae).reshape(-1, len(args.val_srcs)), \
        np.array(val_r2).reshape(-1, len(args.val_srcs))

    print('Program Running Time: %.1f Seconds.' % (time.time() - start_time))

    # Print overall validation results
    print('=' * 80)
    print('Overall Validation Results:\n')

    print('\tBest Results from Different Models (Epochs):')
    # Print best accuracy for cell line classifiers
    clf_targets = ['Cell Line Categories',
                   'Cell Line Sites',
                   'Cell Line Types', ]
    best_acc = np.amax(val_acc, axis=0)
    best_acc_epochs = np.argmax(val_acc, axis=0)

    for index, clf_target in enumerate(clf_targets):
        print('\t\t%24s Best Accuracy: %.3f%% (Epoch = %3d)'
              % (clf_target, best_acc[index], best_acc_epochs[index] + 1))

    # Print best R2 scores for drug response regressor
    val_data_sources = \
        [loader.dataset.data_source for loader in drug_resp_val_loaders]
    best_r2 = np.amax(val_r2, axis=0)
    best_r2_epochs = np.argmax(val_r2, axis=0)

    for index, data_source in enumerate(val_data_sources):
        print('\t\t%6s \t Best R2 Score: %+6.4f '
              '(Epoch = %3d, MSE = %8.2f, MAE = %6.2f)'
              % (data_source, best_r2[index],
                 best_r2_epochs[index] + args.resp_val_start_epoch + 1,
                 val_mse[best_r2_epochs[index], index],
                 val_mae[best_r2_epochs[index], index]))

    # Print best epoch and all the corresponding validation results
    best_epoch = val_r2.sum(axis=1).argmax()
    print('\n\tBest Results from the Same Model (Epoch = %3d):'
          % (best_epoch + 1))
    for index, clf_target in enumerate(clf_targets):
        print('\t\t%24s Best Accuracy: %.3f%%'
              % (clf_target, val_acc[best_epoch, index]))

    for index, data_source in enumerate(val_data_sources):
        print('\t\t%6s \t Best R2 Score: %+6.4f '
              '(MSE = %8.2f, MAE = %6.2f)'
              % (data_source,
                 val_r2[best_epoch, index],
                 val_mse[best_epoch, index],
                 val_mae[best_epoch, index]))


main()
