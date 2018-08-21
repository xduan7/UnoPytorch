""" 
    File Name:          UnoPytorch/weight_initialization.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:   

"""

import torch.nn as nn

# TODO: more complex weight initialization function
# * choice of initialization strategy
# * choice of bias terms
# * choice of different initialization for different types (Linear, Conv, etc.)


def basic_weight_init(module: nn.Module):
    """weight_init(model) or model.apply(weight_init)

    This function initializes the weights of a module using xavier_normal,
    works with leaky ReLU activation function.

    Args:
        module (nn.Module): PyTorch module to be initialized.
    Returns:
        None
    """
    if type(module) in [nn.Linear, ]:
        nn.init.xavier_normal_(module.weight,
                               gain=nn.init.calculate_gain('relu'))
