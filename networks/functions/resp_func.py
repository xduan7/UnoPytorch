""" 
    File Name:          UnoPytorch/resp_func.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:   

"""
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def train_resp(device: torch.device,

               resp_net: nn.Module,
               data_loader: torch.utils.data.DataLoader,

               max_num_batches: int,
               loss_func: callable,
               optimizer: torch.optim, ):

    resp_net.train()
    total_loss = 0.
    num_samples = 0

    for batch_idx, (*ids, rnaseq, drug_feature, conc, grth) \
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


def valid_resp(epoch: int,
               trn_src: str,
               device: torch.device,

               resp_net: nn.Module,
               data_loaders: torch.utils.data.DataLoader,

               resp_uq: bool = False,
               resp_uq_num_runs: int = 100,
               resp_uq_dropout: float = 0.5,

               val_results_dir: str = None):

    if resp_uq:
        resp_net.train()
    else:
        resp_net.eval()

    mse_list = []
    mae_list = []
    r2_list = []

    print('\tDrug Response Regression:')

    resp_net.half()
    with torch.no_grad():
        for val_loader in data_loaders:

            if val_results_dir is not None:
                results_filename = './[trn=%s][val=%s][epoch=%02i].csv' % (
                    trn_src, val_loader.dataset.data_source, epoch + 1)
                results_path = os.path.join(val_results_dir, results_filename)

            results_array = np.array([]).reshape(
                0, 5 + resp_uq_num_runs if resp_uq else 5)

            mse, mae = 0., 0.
            growth_array, pred_array, uq_array = \
                np.array([]), np.array([]), np.array([])

            for drug_id, cell_id, rnaseq, drug_feature, conc, grth \
                    in val_loader:

                rnaseq, drug_feature, conc, grth = \
                    rnaseq.to(device).half(), drug_feature.to(device).half(), \
                    conc.to(device).half(), grth.to(device).half()

                # With uncertainty quantification
                if resp_uq:
                    pred = [
                        resp_net(
                            rnaseq,
                            drug_feature,
                            conc,
                            resp_uq_dropout).view(-1)
                        for _ in range(resp_uq_num_runs)]

                    pred_growth_uq = torch.stack(pred)

                    uq = (pred_growth_uq.var(dim=0)).view(-1, 1)
                    uq_array = np.concatenate(
                        (uq_array, uq.cpu().numpy().flatten()))

                    # pred_growth = pred.mean(dim=0).view(-1, 1)
                    # print(pred_growth.shape)

                # Without uncertainty quantification
                pred_growth = resp_net(rnaseq, drug_feature, conc, dropout=0.)

                # Append the batch results into data structure
                if resp_uq:
                    # temp = np.concatenate(
                    #     (np.array(drug_id).reshape((-1, 1)),
                    #      np.array(cell_id).reshape((-1, 1)),
                    #      conc.cpu().numpy().astype(np.float16),
                    #      grth.cpu().numpy().astype(np.float16),
                    #      pred_growth.cpu().numpy().astype(np.float16),
                    #      torch.t(pred_growth_uq).cpu().numpy().
                    #         astype(np.float16)), axis=1)
                    # print(temp.nbytes)

                    temp = np.concatenate(
                        (np.array(drug_id).reshape((-1, 1)),
                         np.array(cell_id).reshape((-1, 1)),
                         conc.cpu().numpy(),
                         grth.cpu().numpy(),
                         pred_growth.cpu().numpy(),
                         torch.t(pred_growth_uq).cpu().numpy()), axis=1)
                    # print(temp.nbytes)

                else:
                    temp = np.concatenate(
                        (np.array(drug_id).reshape((-1, 1)),
                         np.array(cell_id).reshape((-1, 1)),
                         conc.cpu().numpy(),
                         grth.cpu().numpy(),
                         pred_growth.cpu().numpy()), axis=1)

                results_array = np.concatenate((results_array, temp))
                # print(results_array.nbytes)
                # results_array[start_index: start_index + len(temp), :] = temp
                # start_index += len(temp)

                num_samples = conc.shape[0]
                mse += F.mse_loss(pred_growth, grth).item() * num_samples
                mae += F.l1_loss(pred_growth, grth).item() * num_samples

                growth_array = np.concatenate(
                    (growth_array, grth.cpu().numpy().flatten()))
                pred_array = np.concatenate(
                    (pred_array, pred_growth.cpu().numpy().flatten()))

            # Save the results to file
            col_names = ['drug_id', 'cell_id', 'concentration',
                         'growth', 'predicted_growth', ]
            if resp_uq:
                col_names.extend(
                    ['uq_%03i' % i for i in range(resp_uq_num_runs)])

            if val_results_dir is not None:
                results_dataframe = pd.DataFrame(results_array,
                                                 columns=col_names)
                results_dataframe.to_csv(results_path, index=False)

            # Evaluating validation results
            mse /= len(val_loader.dataset)
            mae /= len(val_loader.dataset)
            r2 = r2_score(y_pred=pred_array, y_true=growth_array)

            mse_list.append(mse)
            mae_list.append(mae)
            r2_list.append(r2)

            print('\t\t%-6s \t MSE: %8.2f \t MAE: %8.2f \t R2: %+4.2f' %
                  (val_loader.dataset.data_source, mse, mae, r2))

            if resp_uq:

                assert len(uq_array) == len(pred_array)

                print('\t\t\tUncertainty Quantification Analysis. \n'
                      '\t\t\tAvg UQ = %8.2f' % uq_array.mean(), )

                num_uq_partitions = 5
                ordered_indices = np.argsort(uq_array)
                partition_size = \
                    int(np.floor(len(uq_array) / num_uq_partitions))

                for i in range(num_uq_partitions):

                    start_index = i * partition_size
                    end_index = (i + 1) * partition_size

                    partitioned_indices = \
                        ordered_indices[start_index: end_index]

                    partitioned_pred_array = \
                        pred_array[partitioned_indices]
                    partitioned_growth_array = \
                        growth_array[partitioned_indices]

                    print('\t\t\tPredictions with UQ [%10.2f, %10.2f]: '
                          ' \t MSE: %8.2f \t MAE: %8.2f \t R2: %+4.2f' %
                          (uq_array[ordered_indices[start_index]],
                           uq_array[ordered_indices[end_index - 1]],
                           mean_squared_error(
                               y_pred=partitioned_pred_array,
                               y_true=partitioned_growth_array),
                           mean_absolute_error(
                               y_pred=partitioned_pred_array,
                               y_true=partitioned_growth_array),
                           r2_score(
                               y_pred=partitioned_pred_array,
                               y_true=partitioned_growth_array)))

    resp_net.float()
    return mse_list, mae_list, r2_list
