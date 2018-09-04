""" 
    File Name:          UnoPytorch/drug_pred_func.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:   

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score


def train_drug_clf(device: torch.device,

                   drug_clf_net: nn.Module,
                   data_loader: torch.utils.data.DataLoader,

                   max_num_batches: int,
                   optimizer: torch.optim, ):

    drug_clf_net.train()

    for batch_idx, (drug_feature, target) in enumerate(data_loader):

        if batch_idx >= max_num_batches:
            break

        drug_feature, target = drug_feature.to(device), target.to(device)

        drug_clf_net.zero_grad()
        out_target = drug_clf_net(drug_feature)
        F.nll_loss(input=out_target, target=target).backward()
        optimizer.step()


def valid_drug_clf(device: torch.device,

                   drug_clf_net: nn.Module,
                   data_loader: torch.utils.data.DataLoader, ):

    drug_clf_net.eval()

    correct_target = 0

    with torch.no_grad():
        for drug_feature, target in data_loader:

            drug_feature, target = drug_feature.to(device), target.to(device)

            out_target = drug_clf_net(drug_feature)
            pred_target = out_target.max(1, keepdim=True)[1]

            correct_target += pred_target.eq(
                target.view_as(pred_target)).sum().item()

    # Get overall accuracy
    target_acc = 100. * correct_target / len(data_loader.dataset)

    print('\tDrug Target Family Classification Accuracy: %5.2f%%' % target_acc)

    return target_acc


def train_drug_rgs(device: torch.device,

                   drug_rgs_net: nn.Module,
                   data_loader: torch.utils.data.DataLoader,

                   max_num_batches: int,
                   loss_func: callable,
                   optimizer: torch.optim, ):

    drug_rgs_net.train()
    total_loss = 0.
    num_samples = 0

    for batch_idx, (drug_feature, target) in enumerate(data_loader):

        if batch_idx >= max_num_batches:
            break

        drug_feature, target = drug_feature.to(device), target.to(device)

        drug_rgs_net.zero_grad()
        pred_target = drug_rgs_net(drug_feature)

        loss = loss_func(pred_target, target)

        loss.backward()
        optimizer.step()

        num_samples += target.shape[0]
        total_loss += loss.item() * target.shape[0]

    print('\tDrug Weighted QED Regression Loss: %8.2f'
          % (total_loss / num_samples))


def valid_drug_rgs(device: torch.device,

                   drug_rgs_net: nn.Module,

                   data_loader: torch.utils.data.DataLoader, ):

    drug_rgs_net.eval()
    mse, mae = 0., 0.
    target_array, pred_array = np.array([]), np.array([])

    print('\tDrug Weighted QED Regression:')

    with torch.no_grad():
        for drug_feature, target in data_loader:

            drug_feature, target = drug_feature.to(device), target.to(device)

            pred_target = drug_rgs_net(drug_feature)

            num_samples = target.shape[0]
            mse += F.mse_loss(pred_target, target).item() * num_samples
            mae += F.l1_loss(pred_target, target).item() * num_samples

            target_array = np.concatenate(
                (target_array, target.cpu().numpy().flatten()))
            pred_array = np.concatenate(
                (pred_array, pred_target.cpu().numpy().flatten()))

        mse /= len(data_loader.dataset)
        mae /= len(data_loader.dataset)
        r2 = r2_score(y_pred=pred_array, y_true=data_loader)

    print('\tDrug Weighted QED Regression '
          '\t MSE: %8.2f \t MAE: %8.2f \t R2: %+4.2f' % (mse, mae, r2))

    return mse, mae, r2
