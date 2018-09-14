""" 
    File Name:          batch_prediction.py 
    Project Name:       UnoPytorch
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/14/2018
    Python Version:     3.6.4
    File Description:   

"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from networks.functions.cl_clf_func import train_cl_clf, valid_cl_clf
from networks.functions.drug_qed_func import train_drug_qed, valid_drug_qed
from networks.functions.drug_target_func import train_drug_target, \
    valid_drug_target
from networks.functions.resp_func import train_resp, valid_resp
from networks.initialization.weight_init import basic_weight_init
from networks.structures.classification_net import ClfNet
from networks.structures.regression_net import RgsNet
from networks.structures.residual_block import ResBlock
from networks.structures.response_net import RespNet
from utils.data_processing.label_encoding import get_label_dict
from utils.datasets.drug_qed_dataset import DrugQEDDataset
from utils.datasets.drug_resp_dataset import DrugRespDataset
from utils.datasets.cl_class_dataset import CLClassDataset
from utils.data_processing.dataframe_scaling import SCALING_METHODS
from networks.initialization.encoder_init import get_gene_encoder, \
    get_drug_encoder
from utils.datasets.drug_target_dataset import DrugTargetDataset
from utils.miscellaneous.optimizer import get_optimizer
from utils.miscellaneous.random_seeding import seed_random_state


# Number of workers for dataloader. Too many workers might lead to process
# hanging for PyTorch version 4.1. Set this number between 0 and 4.
NUM_WORKER = 4
DATA_ROOT = './data/'


def main():
    # Training settings and hyper-parameters
    parser = argparse.ArgumentParser(
        description='Data Source (Batch) Prediction for Cell Lines')

    # Dataset parameters ######################################################
    # Pre-processing for dataframes
    parser.add_argument('--rnaseq_scaling', type=str, default='std',
                        help='scaling method for RNA sequence',
                        choices=SCALING_METHODS)

    # Feature usage and partitioning settings
    parser.add_argument('--rnaseq_feature_usage', type=str,
                        default='combat',
                        help='RNA sequence data used',
                        choices=['source_scale', 'combat', ])
    parser.add_argument('--validation_ratio', type=float, default=0.2,
                        help='ratio for validation dataset')

    # Network configuration ###################################################
    parser.add_argument('--layer_dim', type=int, default=256,
                        help='dimension of layers for RNA sequence')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of layers for RNA sequence')

    # Training and validation parameters ######################################
    parser.add_argument('--opt', type=str, default='SGD',
                        help='optimizer for data source prediction',
                        choices=['SGD', 'RMSprop', 'Adam'])
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate for data source prediction')

    # Starting epoch for validation
    parser.add_argument('--val_start_epoch', type=int, default=0,
                        help='starting epoch for data source prediction')

    # Early stopping based on data source prediction accuracy
    parser.add_argument('--early_stop_patience', type=int, default=50,
                        help='patience for early stopping based on data '
                             'source prediction accuracy')

    # Global/shared training parameters
    parser.add_argument('--l2_regularization', type=float, default=0.,
                        help='L2 regularization for nn weights')
    parser.add_argument('--lr_decay_factor', type=float, default=0.98,
                        help='decay factor for learning rate')
    parser.add_argument('--trn_batch_size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help='input batch size for validation')
    parser.add_argument('--max_num_batches', type=int, default=10000,
                        help='maximum number of batches per epoch')
    parser.add_argument('--max_num_epochs', type=int, default=1000,
                        help='maximum number of epochs')

    # Miscellaneous settings ##################################################
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
    cl_clf_dataset_kwargs = {
        'data_root': DATA_ROOT,
        'rand_state': args.rand_state,
        'summary': False,

        'int_dtype': np.int8,
        'float_dtype': np.float16,
        'output_dtype': np.float32,

        'rnaseq_scaling': args.rnaseq_scaling,
        'predict_target': 'source',

        'rnaseq_feature_usage': args.rnaseq_feature_usage,
        'validation_ratio': args.validation_ratio, }

    cl_clf_trn_loader = torch.utils.data.DataLoader(
        CLClassDataset(training=True,
                       **cl_clf_dataset_kwargs),
        batch_size=args.trn_batch_size,
        **dataloader_kwargs)

    cl_clf_val_loader = torch.utils.data.DataLoader(
        CLClassDataset(training=False,
                       **cl_clf_dataset_kwargs),
        batch_size=args.val_batch_size,
        **dataloader_kwargs)

    # Constructing and initializing neural networks ###########################
    net = nn.Sequential()

    prev_dim = cl_clf_trn_loader.dataset.rnaseq_dim
    for label in ['site', 'type', 'category']:
        prev_dim += len(get_label_dict(DATA_ROOT, '%s_dict.txt' % label))

    # net.add_module('dense_%d' % 0, nn.Linear(prev_dim, args.layer_dim))

    for i in range(args.num_layers):
        # net.add_module('residual_block_%d' % i,
        #                ResBlock(layer_dim=args.layer_dim,
        #                         num_layers=2,
        #                         dropout=0.))

        net.add_module('dense_%d' % i, nn.Linear(prev_dim, args.layer_dim))
        net.add_module('dropout_%d' % i, nn.Dropout(0.2))
        prev_dim = args.layer_dim
        net.add_module('relu_%d' % i, nn.ReLU())

    num_data_src = len(get_label_dict(DATA_ROOT, 'data_src_dict.txt'))
    net.add_module('dense', nn.Linear(args.layer_dim, num_data_src))
    net.add_module('logsoftmax', nn.LogSoftmax(dim=1))
    net.apply(basic_weight_init)
    net.to(device)

    print(net)

    # Optimizers, learning rate decay, and miscellaneous ######################
    opt = get_optimizer(opt_type=args.opt,
                        networks=net,
                        learning_rate=args.lr,
                        l2_regularization=args.l2_regularization)
    lr_decay = LambdaLR(optimizer=opt,
                        lr_lambda=lambda e:
                        args.lr_decay_factor ** e)

    # Training/validation loops ###############################################
    val_acc = []
    best_acc = 0.
    patience = 0
    start_time = time.time()

    for epoch in range(args.max_num_epochs):

        print('=' * 80 + '\nTraining Epoch %3i:' % (epoch + 1))
        epoch_start_time = time.time()

        lr_decay.step(epoch)

        # Training loop #######################################################
        net.train()

        for batch_idx, (rnaseq, data_src, cl_site, cl_type, cl_category) \
                in enumerate(cl_clf_trn_loader):

            if batch_idx >= args.max_num_batches:
                break

            rnaseq, data_src, cl_site, cl_type, cl_category = \
                rnaseq.to(device), data_src.to(device), cl_site.to(device), \
                cl_type.to(device), cl_category.to(device)

            net.zero_grad()

            out_data_src = net(torch.cat(
                (rnaseq, cl_site, cl_type, cl_category), dim=1))

            F.nll_loss(input=out_data_src, target=data_src).backward()

            opt.step()

        # Validation loop #####################################################
        net.eval()

        correct_data_src = 0
        with torch.no_grad():
            for rnaseq, data_src, cl_site, cl_type, cl_category \
                    in cl_clf_val_loader:

                rnaseq, data_src, cl_site, cl_type, cl_category = \
                    rnaseq.to(device), data_src.to(device), \
                    cl_site.to(device), cl_type.to(device), \
                    cl_category.to(device)

                out_data_src = net(torch.cat(
                    (rnaseq, cl_site, cl_type, cl_category), dim=1))

                pred_data_src = out_data_src.max(1, keepdim=True)[1]

                # print(data_src)
                # print(pred_data_src)

                correct_data_src += pred_data_src.eq(
                    data_src.view_as(pred_data_src)).sum().item()

        data_src_acc = 100. * correct_data_src / len(cl_clf_val_loader.dataset)

        print('\tCell Line Data Source (Batch) Prediction Accuracy: %5.2f%%; '
              % data_src_acc)

        # Results recording and early stopping
        val_acc.append(data_src_acc)

        if data_src_acc > best_acc:
            patience = 0
            best_acc = data_src_acc
        else:
            patience += 1
        if patience >= args.early_stop_patience:
            print('Validation accuracy does not improve for %d epochs ... '
                  'invoking early stopping.' % patience)
            break

        print('Epoch Running Time: %.1f Seconds.'
              % (time.time() - epoch_start_time))

    print('Program Running Time: %.1f Seconds.' % (time.time() - start_time))
    print('Best Cell Line Data Source (Batch) Prediction Accuracy: %5.2f%%; '
          % np.amax(val_acc))


if __name__ == '__main__':
    main()
