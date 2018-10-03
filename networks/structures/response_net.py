""" 
    File Name:          UnoPytorch/response_net.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/17/18
    Python Version:     3.6.6
    File Description:   

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.structures.residual_block import ResBlock
from networks.initialization.weight_init import basic_weight_init


class RespNet(nn.Module):

    def __init__(self,
                 gene_latent_dim: int,
                 drug_latent_dim: int,

                 gene_encoder: nn.Module,
                 drug_encoder: nn.Module,

                 resp_layer_dim: int,
                 resp_num_layers_per_block: int,
                 resp_num_blocks: int,

                 resp_num_layers: int,

                 resp_dropout: float,
                 resp_activation: str):

        super(RespNet, self).__init__()

        self.__gene_encoder = gene_encoder
        self.__drug_encoder = drug_encoder

        # Layer construction ##################################################
        self.__dropout = resp_dropout
        self.__activation = resp_activation
        self.total_num_layers = \
            2 + resp_num_layers_per_block * resp_num_blocks + resp_num_layers

        self.__resp_net = nn.ModuleList([])
        self.__resp_net.append(
            nn.Linear(gene_latent_dim + drug_latent_dim + 1, resp_layer_dim))

        for i in range(resp_num_blocks):
            self.__resp_net.append(
                ResBlock(layer_dim=resp_layer_dim,
                         num_layers=resp_num_layers_per_block,
                         dropout=resp_dropout))

        for i in range(resp_num_layers):
            self.__resp_net.append(nn.Linear(resp_layer_dim, resp_layer_dim))

        self.__resp_net.append(nn.Linear(resp_layer_dim, 1))

        # Weight Initialization ###############################################
        self.__resp_net.apply(basic_weight_init)

    def forward(self, rnaseq, drug_feature, concentration, dropout=None):

        if dropout is None:
            p = self.__dropout
        else:
            if not self.training:
                raise ValueError('Testing mode with specified dropout rate')
            p = dropout

        x = torch.cat((self.__gene_encoder(rnaseq),
                       self.__drug_encoder(drug_feature),
                       concentration), dim=1)

        for i, layer in enumerate(self.__resp_net):

            if type(layer) == ResBlock:
                x = layer(x, dropout)
            else:
                x = F.dropout(x, p=p, training=self.training)
                x = layer(x)

                if i < len(self.__resp_net) - 1:
                    x = F.relu(x)
                else:
                    if self.__activation.lower() == 'sigmoid':
                        x = F.sigmoid(x)
                    elif self.__activation.lower() == 'tanh':
                        x = F.tanh(x)
                    else:
                        pass

        return x
