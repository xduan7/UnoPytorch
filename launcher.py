""" 
    File Name:          UnoPytorch/launcher.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/17/18
    Python Version:     3.6.6
    File Description:   

"""

import sys
import runpy
import datetime

from utils.miscellaneous.tee import Tee

if __name__ == '__main__':

    param_dict_list = [

        {'trn_src': ['GDSC'],
         'val_srcs': ['GDSC', 'CCLE', 'gCSI'], },

        # Training + validation data sources for the transfer learning matrix
        {'trn_src': ['NCI60'],
         'val_srcs': ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'], },

        {'trn_src': ['CTRP'],
         'val_srcs': ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'], },

        {'trn_src': ['GDSC'],
         'val_srcs': ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'], },

        {'trn_src': ['CCLE'],
         'val_srcs': ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'], },

        {'trn_src': ['gCSI'],
         'val_srcs': ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'], },
    ]

    # for rnaseq_feature in ['source_scale', 'combat']:
    rnaseq_feature = 'source_scale'

    # for resp_lr, resp_loss in zip(['1e-5'], ['mse']):

    for param_dict in param_dict_list:

        # Try until the models are successfully trained

        # module_finished = False
        # while not module_finished:

        now = datetime.datetime.now()

        # Save log with name = (training data source + time)
        tee = Tee('./results/logs/' + rnaseq_feature +
                  '/%s_(%02d%02d_%02d%02d).txt'
                  % (param_dict['trn_src'],
                     now.month, now.day, now.hour, now.minute))
        sys.stdout = tee

        sys.argv = [
            'uno_pytorch',

            # Dataset parameters ######################################
            # Training and validation data sources
            '--trn_src', *param_dict['trn_src'],
            '--val_srcs', *param_dict['val_srcs'],

            # Pre-processing for dataframes
            '--growth_scaling', 'none',
            '--descriptor_scaling', 'std',
            '--rnaseq_scaling', 'std',
            '--descriptor_nan_threshold', '0.0',

            # Feature usage and partitioning settings
            '--rnaseq_feature_usage', rnaseq_feature,
            '--drug_feature_usage', 'both',
            '--validation_ratio', '0.15',
            # '--disjoint_drugs',
            '--disjoint_cells',

            # Network configuration ###################################
            # Encoders for drug features and RNA sequence
            '--gene_layer_dim', '1024',
            '--gene_num_layers', '2',
            '--gene_latent_dim', '512',

            '--drug_layer_dim', '4096',
            '--drug_num_layers', '2',
            '--drug_latent_dim', '1024',

            # Using autoencoder for drug/sequence encoder init
            '--autoencoder_init',

            # Drug response regression network
            '--resp_layer_dim', '2048',
            '--resp_num_layers_per_block', '2',
            '--resp_num_blocks', '2',
            '--resp_num_layers', '2',
            '--resp_dropout', '0.5',
            '--resp_activation', 'none',

            # RNA sequence classification network(s)
            '--clf_layer_dim', '256',
            '--clf_num_layers', '2',

            # Training and validation parameters ######################
            # Drug response regression training parameters
            '--resp_loss_func', 'mse',
            '--resp_opt', 'SGD',
            '--resp_lr', '1e-5',

            # Starting epoch for drug response validation
            '--resp_val_start_epoch', '0',

            # Early stopping based on R2 score of drug response
            '--early_stop_patience', '20',

            # RNA sequence classification training parameters
            '--clf_opt', 'SGD',
            '--clf_lr', '1e-3',

            # Global/shared training parameters
            '--lr_decay_factor', '0.98',
            '--trn_batch_size', '32',
            '--val_batch_size', '256',
            '--max_num_batches', '1000',
            '--max_num_epochs', '3',

            # Miscellaneous settings ##################################
            # '--multi_gpu'
            # '--no_cuda'
            '--rand_state', '0', ]

        runpy.run_module('uno_pytorch')

        # module_finished = True
        # except Exception as e:
        #     print('Encountering Exception %s' % e)
        #     print('Re-initiate a new run ...')

        sys.stdout = tee.default_stdout()
