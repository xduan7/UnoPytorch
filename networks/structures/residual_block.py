""" 
    File Name:          UnoPytorch/residual_block.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:   

"""
import torch.nn as nn
import torch.nn.functional as F
from networks.initialization.weight_init import basic_weight_init


class ResBlock(nn.Module):

    def __init__(self,

                 layer_dim: int,
                 num_layers: int,

                 dropout: float):

        super(ResBlock, self).__init__()

        self.__dropout = dropout

        # Layer construction ##################################################
        self.__layers = nn.ModuleList(
            [nn.Linear(layer_dim, layer_dim) for _ in range(num_layers)])

        # Weight Initialization ###############################################
        self.apply(basic_weight_init)

    def forward(self, x, dropout=None):

        if dropout is None:
            p = self.__dropout
        else:
            if not self.training:
                raise ValueError('Testing mode with specified dropout rate')
            p = dropout

        x0 = None

        for i, layer in enumerate(self.__layers):

            x = F.dropout(x, p=p, training=self.training)
            x0 = x if i == 0 else x0
            x = layer(x)
            x = F.relu(x)

        assert x0 is not None

        return x0 + x


if __name__ == '__main__':

    res_block = ResBlock(
        layer_dim=200,
        num_layers=2,
        dropout=0.2)

    print(res_block)
